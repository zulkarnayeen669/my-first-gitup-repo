"""
Handcrafted features for the classical ML baselines of Table III
(SVM-RBF, Random Forest, k-NN (k=5), Logistic Regression).

The paper does not state which inputs the classical models received; here they
get the same fused (sensor, spiral, video) triples as the proposed model, each
summarised by standard tremor features, so all rows of Table III are evaluated
on identical test samples.
"""
import numpy as np
from scipy.signal import welch

FS_SENSOR = 50.0  # after decimation in load_sensor
FS_VIDEO = 15.0
BANDS = [(2, 4), (4, 6.5), (6.5, 9), (9, 13), (13, 20), (20, 25)]


def sensor_features(X):
    """X: N x T x 6 -> time stats + relative Welch band power per channel."""
    f, P = welch(X, fs=FS_SENSOR, nperseg=64, axis=1)          # N x F x 6
    tot = P.sum(1) + 1e-12
    bp = [P[:, (f >= a) & (f < b)].sum(1) / tot for a, b in BANDS]
    peak = f[P[:, f > 2].argmax(1) + (f <= 2).sum()]           # dominant freq > 2 Hz
    stats = [X.std(1), np.abs(np.diff(X, axis=1)).mean(1),
             ((X[:, 1:] * X[:, :-1]) < 0).mean(1)]             # zero-crossing rate
    return np.concatenate(bp + [peak] + stats, 1)


def spiral_features(X):
    """X: N x 1 x 64 x 64 (ink=1) -> ink statistics, radial profile, roughness."""
    img = X[:, 0]
    H = img.shape[-1]
    yy, xx = np.mgrid[0:H, 0:H]
    r = np.hypot(yy - H / 2, xx - H / 2)
    out = []
    for im in img:
        ink = im.sum() + 1e-6
        cy, cx = (im * yy).sum() / ink, (im * xx).sum() / ink
        rad = np.bincount(r.astype(int).ravel(), im.ravel(), minlength=46)[:46]
        gy, gx = np.gradient(im)
        g = np.hypot(gx, gy)
        lap = np.abs(np.gradient(gx, axis=1) + np.gradient(gy, axis=0))
        out.append(np.r_[ink / im.size, im.max(), (im > 0.3).mean(), cy - H / 2, cx - H / 2,
                         g.mean(), g.std(), lap.mean(), lap.std(),
                         (r * im).sum() / ink, rad / (rad.sum() + 1e-6),
                         np.abs(np.fft.fft2(im))[:8, :8].ravel() / ink])
    return np.array(out)


def video_features(X):
    """X: N x 1 x T x H x W -> motion energy and temporal spectrum of the
    hand-region intensity signal."""
    v = X[:, 0]
    d = np.diff(v, axis=1)
    me = np.abs(d).mean((2, 3))                                 # N x (T-1)
    sd = v.std(1)                                               # temporal std map
    thr = np.quantile(sd.reshape(len(v), -1), 0.9, axis=1)
    frac = (sd > thr[:, None, None] * 0.8).mean((1, 2))
    # centroid trajectory of the bright (hand) region
    H = v.shape[-1]
    yy, xx = np.mgrid[0:H, 0:H]
    w = np.clip(v, 0, None)
    m = w.sum((2, 3)) + 1e-6
    cy, cx = (w * yy).sum((2, 3)) / m, (w * xx).sum((2, 3)) / m
    spec = [np.abs(np.fft.rfft(c - c.mean(1, keepdims=True), axis=1))[:, 1:] for c in (cy, cx, me)]
    return np.concatenate([me.mean(1, keepdims=True), me.std(1, keepdims=True), frac[:, None],
                           sd.mean((1, 2))[:, None], sd.max((1, 2))[:, None],
                           cy.std(1, keepdims=True), cx.std(1, keepdims=True)] + spec, 1)
