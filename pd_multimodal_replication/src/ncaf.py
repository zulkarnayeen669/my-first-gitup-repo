"""
Extension (not part of Chandra et al. 2026): cross-modal attention fusion
following S. Gujjeti et al., "NeuroCrossAttention fusion for multimodal
explainable early diagnosis of neurodegenerative diseases", Discover
Computing 29:329, 2026 (NCAF), adapted to the three hand-tremor modalities.

  tokens t_m = LayerNorm(W_m Z_m) + e_m           m in {sensor, spiral, video}
  alpha_ij  = softmax_j(Q_i K_j^T / sqrt(d))      (NCAF Eq. 3, multi-head)
  H_i       = sum_j alpha_ij V_j                  (NCAF Eq. 4)
  F         = [H_sensor || H_spiral || H_video]   (NCAF Eq. 5)
  y_hat     = Softmax(W_o Dropout(ReLU(W F + b)))

Modality masking (NCAF Sec. 4.2): during training each modality is dropped
with probability p_mask (at least one is always kept); a dropped modality's
token is zeroed and excluded as a key, so at test time the model can run
with any subset of modalities. The attention weights of the last forward
pass are stored in `last_alpha` (B x 3 x 3, averaged over heads) for the
explainability module.

Also here: ConcatMLPHead (plain concatenation fusion) and HeadOnEmbeddings,
which adapts the paper's GRU-LSTNet + BAM head to the same interface so all
fusion heads can be compared on identical frozen embeddings.
"""
import torch
import torch.nn as nn

DIMS = (32, 64, 64)  # Z_sensor, Z_spiral, Z_video


def _sample_present(B, n_mod, p_mask, device):
    present = torch.rand(B, n_mod, device=device) >= p_mask
    none = ~present.any(1)
    if none.any():  # keep one random modality for samples where all were dropped
        keep = torch.randint(0, n_mod, (int(none.sum()),), device=device)
        present[none.nonzero(as_tuple=True)[0], keep] = True
    return present


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, dims=DIMS, d=64, heads=4, hidden=128, p_drop=0.5, p_mask=0.0):
        super().__init__()
        self.n, self.d, self.h, self.p_mask = len(dims), d, heads, p_mask
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(k, d), nn.LayerNorm(d)) for k in dims])
        self.mod_emb = nn.Parameter(torch.randn(self.n, d) * 0.02)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.fc = nn.Sequential(nn.Linear(self.n * d, hidden), nn.ReLU(), nn.Dropout(p_drop))
        self.out = nn.Linear(hidden, 2)
        self.last_alpha = None

    def forward(self, zs, present=None):
        B = zs[0].shape[0]
        t = torch.stack([p(z) for p, z in zip(self.proj, zs)], 1) + self.mod_emb      # B x 3 x d
        if present is None:
            present = (_sample_present(B, self.n, self.p_mask, t.device) if self.training and self.p_mask > 0
                       else torch.ones(B, self.n, dtype=torch.bool, device=t.device))
        t = t * present[..., None]
        dh = self.d // self.h
        Q, K, V = (f(t).view(B, self.n, self.h, dh).transpose(1, 2) for f in (self.q, self.k, self.v))
        scores = Q @ K.transpose(-1, -2) / dh ** 0.5                                   # B x h x 3 x 3
        scores = scores.masked_fill(~present[:, None, None, :], float("-inf"))
        alpha = torch.softmax(scores, -1)                                               # Eq. 3
        H = (alpha @ V).transpose(1, 2).reshape(B, self.n, self.d)                      # Eq. 4
        self.last_alpha = alpha.mean(1)
        return self.out(self.fc(H.flatten(1)))                                          # Eq. 5 + FC


class ConcatMLPHead(nn.Module):
    """Plain feature concatenation -> FC -> softmax. Missing modalities are zeroed."""

    def __init__(self, dims=DIMS, hidden=128, p_drop=0.5, p_mask=0.0):
        super().__init__()
        self.n, self.p_mask = len(dims), p_mask
        self.net = nn.Sequential(nn.Linear(sum(dims), hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, 2))

    def forward(self, zs, present=None):
        if present is None and self.training and self.p_mask > 0:
            present = _sample_present(zs[0].shape[0], self.n, self.p_mask, zs[0].device)
        if present is not None:
            zs = [z * present[:, i:i + 1] for i, z in enumerate(zs)]
        return self.net(torch.cat(zs, 1))


class HeadOnEmbeddings(nn.Module):
    """Wraps a head taking Z_fused (B x 160), e.g. the paper's HybridGRULSTNetBAM."""

    def __init__(self, head, n=3):
        super().__init__()
        self.head, self.n = head, n

    def forward(self, zs, present=None):
        if present is not None:
            zs = [z * present[:, i:i + 1] for i, z in enumerate(zs)]
        return self.head(torch.cat(zs, 1))


class EncodersPlusHead(nn.Module):
    """Full model: raw (sensor, spiral, video) -> encoders -> fusion head.
    Used by the explainability module so gradients reach the raw inputs."""

    def __init__(self, sensor, spiral, video, head):
        super().__init__()
        self.sensor, self.spiral, self.video, self.head = sensor, spiral, video, head

    def forward(self, xs, xi, xv, present=None):
        return self.head([self.sensor(xs), self.spiral(xi), self.video(xv)], present)
