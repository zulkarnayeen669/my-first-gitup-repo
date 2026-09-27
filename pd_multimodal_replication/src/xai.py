"""
Explainability module (extension, after Gujjeti et al. 2026, Sec. 4.4 /
Algorithm 3), adapted to the tremor model. All functions take the full
`EncodersPlusHead` model so explanations refer to raw inputs.

  global   attention_matrix         mean cross-modal attention alpha_ij (NCAF Eq. 3)
  local    embedding_saliency       eta = d y_hat / d Z_m, as |grad x input| per modality (NCAF Eq. 6)
           hybrid_modality_scores   E = lambda*A + (1-lambda)*S (NCAF Eq. 7), per patient
           gradcam_2d / gradcam_3d  Grad-CAM on the last conv block of the spiral / video encoder
           frequency_occlusion      p(PD) change when a frequency band is removed from the sensor window
  checks   modality_deletion        faithfulness: drop the top- vs bottom-ranked modality
           cam_localisation         share of Grad-CAM mass on the ink / moving-hand region vs its area

Unlike the NCAF paper, whose explanation checks are qualitative, the
synthetic data here has a known ground truth (where the tremor is and at
which frequency), so every check returns a number.
"""
import numpy as np
import torch
import torch.nn.functional as F

MODS = ["sensor", "spiral", "video"]


def _p_pd(logits):
    return torch.softmax(logits, 1)[:, 1]


@torch.no_grad()
def attention_matrix(model, xs, xi, xv, bs=64):
    """Per-sample alpha (N x 3 x 3), query modality i (rows) -> key modality j (cols)."""
    model.eval()
    out = []
    for i in range(0, len(xs), bs):
        model(xs[i:i + bs], xi[i:i + bs], xv[i:i + bs])
        out.append(model.head.last_alpha.clone())
    return torch.cat(out).numpy()


def embedding_saliency(model, xs, xi, xv, bs=64):
    """S_m = sum |Z_m * d p(PD) / d Z_m| for every sample (N x 3)."""
    model.eval()
    out = []
    for i in range(0, len(xs), bs):
        with torch.no_grad():
            zs = [model.sensor(xs[i:i + bs]), model.spiral(xi[i:i + bs]), model.video(xv[i:i + bs])]
        zs = [z.clone().requires_grad_(True) for z in zs]
        p = _p_pd(model.head(zs))
        g = torch.autograd.grad(p.sum(), zs)
        out.append(torch.stack([(z * gz).abs().sum(1) for z, gz in zip(zs, g)], 1).detach())
    return torch.cat(out).numpy()


def hybrid_modality_scores(alpha, sal, lam=0.5):
    """A_j = attention received by modality j (mean over queries), S = saliency;
    both normalised to sum to 1 per patient; E = lam*A + (1-lam)*S."""
    A = alpha.mean(1)
    A = A / A.sum(1, keepdims=True)
    S = sal / (sal.sum(1, keepdims=True) + 1e-12)
    return lam * A + (1 - lam) * S, A, S


@torch.no_grad()
def modality_deletion(model, xs, xi, xv, scores, bs=64):
    """Faithfulness test: remove (mask) the modality with the highest vs the
    lowest explanation score; a faithful explanation makes the first hurt more.
    Returns mean |delta p(PD)| for (top, bottom)."""
    model.eval()
    d_top, d_bot = [], []
    for i in range(0, len(xs), bs):
        sl = slice(i, i + bs)
        base = _p_pd(model(xs[sl], xi[sl], xv[sl]))
        for which, store in ((scores[sl].argmax(1), d_top), (scores[sl].argmin(1), d_bot)):
            present = torch.ones(len(base), 3, dtype=torch.bool)
            present[torch.arange(len(base)), torch.as_tensor(which)] = False
            store.append((base - _p_pd(model(xs[sl], xi[sl], xv[sl], present))).abs())
    return torch.cat(d_top).mean().item(), torch.cat(d_bot).mean().item()


def _gradcam(model, layer, xs, xi, xv, spatial_dims):
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["a"] = out

    h = layer.register_forward_hook(hook)
    model.eval()
    p = _p_pd(model(xs, xi, xv))
    target = torch.where(p >= 0.5, p, 1 - p)   # explain the predicted class
    model.zero_grad()
    target.sum().backward()
    h.remove()
    a, g = acts["a"], acts["a"].grad
    w = g.mean(dim=spatial_dims, keepdim=True)
    cam = torch.relu((w * a).sum(1))
    return cam.detach(), p.detach()


def gradcam_2d(model, xs, xi, xv, layer=None):
    """Grad-CAM for the predicted class on the spiral encoder's last conv block, upsampled to 64x64."""
    layer = layer or model.spiral.features[2][1]           # ReLU of the 3rd conv block (16x16)
    cam, p = _gradcam(model, layer, xs, xi, xv, (2, 3))
    cam = F.interpolate(cam[:, None], size=xi.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.numpy(), p.numpy()


def gradcam_3d(model, xs, xi, xv, layer=None):
    """Grad-CAM for the predicted class on the video encoder's last conv block, upsampled to T x 64 x 64."""
    layer = layer or model.video.features[7]               # ReLU of the 3rd Conv3d (8x16x16)
    cam, p = _gradcam(model, layer, xs, xi, xv, (2, 3, 4))
    cam = F.interpolate(cam[:, None], size=xv.shape[-3:], mode="trilinear", align_corners=False)[:, 0]
    return cam.numpy(), p.numpy()


def cam_localisation(cam, mask):
    """(share of CAM mass inside mask, share of area covered by mask). A ratio > 1
    means the explanation concentrates on the region of interest."""
    cam = cam.reshape(len(cam), -1)
    mask = mask.reshape(len(mask), -1).astype(bool)
    inside = (cam * mask).sum(1) / (cam.sum(1) + 1e-12)
    return inside, mask.mean(1)


@torch.no_grad()
def frequency_occlusion(model, xs, xi, xv, bands, fs=50.0, present=None, bs=64):
    """For each band, zero its FFT bins in the sensor window (all channels) and
    record p(PD) after - before. Returns (N x n_bands) deltas and base p."""
    model.eval()
    T = xs.shape[1]
    f = np.fft.rfftfreq(T, 1 / fs)
    X = torch.fft.rfft(xs, dim=1)
    base = torch.cat([_p_pd(model(xs[i:i + bs], xi[i:i + bs], xv[i:i + bs],
                                  None if present is None else present[i:i + bs]))
                      for i in range(0, len(xs), bs)])
    out = []
    for a, b in bands:
        keep = torch.as_tensor(~((f >= a) & (f < b)), dtype=X.dtype)[None, :, None]
        xs_b = torch.fft.irfft(X * keep, n=T, dim=1)
        pb = torch.cat([_p_pd(model(xs_b[i:i + bs], xi[i:i + bs], xv[i:i + bs],
                                    None if present is None else present[i:i + bs]))
                        for i in range(0, len(xs), bs)])
        out.append((pb - base).numpy())
    return np.stack(out, 1), base.numpy()


def time_saliency(model, xs, xi, xv, present=None):
    """Gradient x input of p(PD) w.r.t. the raw sensor window (N x T x 6)."""
    model.eval()
    xs = xs.clone().requires_grad_(True)
    p = _p_pd(model(xs, xi, xv, present))
    (g,) = torch.autograd.grad(p.sum(), xs)
    return (g * xs).detach().numpy()
