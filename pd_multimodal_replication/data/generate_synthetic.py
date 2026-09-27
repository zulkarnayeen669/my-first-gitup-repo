"""
Synthetic stand-ins for the three public datasets used by
Chandra, Gandhi & Singh, "A Hybrid Multimodal Deep Learning Framework for
Robust Diagnosis of Parkinson's Disease Using Hand Tremor Analysis",
IEEE Sensors Journal 26(3), 2026 (Table I).

    Paper dataset                        -> file layout written here (same shape as the real one)
    -----------------------------------------------------------------------------------------------
    MPU-9250 kinematic sensor dataset    -> synthetic/sensor/<subject>.csv.gz
        tri-axial acc + gyro (+mag) CSV     columns: t, ax, ay, az, gx, gy, gz, mx, my, mz
    Dryad Parkinson's Drawing dataset    -> synthetic/spiral/{healthy,parkinson}/<subject>_<k>.png
        digitised spiral images             (Kaggle "parkinsons-drawings"-style folders)
    PD Motor Severity (video) dataset    -> synthetic/video/clips.npz
        RGB hand-motion videos              uint8 array (N, 16, 64, 64) + labels + subject ids

Every subject gets a clinical label (healthy / PD) and, for PD, a tremor
severity score s in {0..4} (MDS-UPDRS item 3.15/3.17 style). The signals are
generated from simple physiological models:

  * Healthy: physiological tremor 8-12 Hz, very small amplitude; ~25% of
    controls have "enhanced physiological tremor" (5.5-9 Hz, larger).
  * PD: rest tremor 3.5-7 Hz whose amplitude grows with s (s = 0 means the
    subject has no visible tremor, i.e. is indistinguishable from a control),
    amplitude waxing/waning, bradykinesia (reduced voluntary movement),
    micrographia and lower/less stable pen pressure in drawings.

This deliberately yields an irreducible error of roughly 10-20% per modality,
so the task is not trivially separable. Everything is seeded and reproducible.

Usage:  python data/generate_synthetic.py [--out data/synthetic] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

FS = 100  # Hz, sensor and pen sampling rate
# Domain shift for an "external cohort" (different device / site). 0 = the
# main cohort. It changes only deterministic factors (no extra random draws),
# so SHIFT = 0 reproduces the main dataset exactly.
SHIFT = 0.0
SEVERITY_P = [0.15, 0.30, 0.30, 0.15, 0.10]  # P(s = 0..4) among PD subjects


def make_subjects(n_pd, n_hc, prefix, rng):
    rows = []
    for i in range(n_hc):
        rows.append(dict(subject=f"{prefix}HC{i:03d}", label=0, severity=0,
                         enhanced=bool(rng.random() < 0.25)))
    for i in range(n_pd):
        rows.append(dict(subject=f"{prefix}PD{i:03d}", label=1,
                         severity=int(rng.choice(5, p=SEVERITY_P)), enhanced=False))
    return rows


def tremor_params(subj, rng):
    """(frequency Hz, amplitude a.u.) of the dominant tremor of a subject."""
    if subj["label"] == 1:
        s = subj["severity"]
        f = rng.uniform(3.5, 7.0)
        amp = 0.25 if s == 0 else 0.6 * s ** 1.2
        amp *= rng.lognormal(0, 0.35)
    elif subj["enhanced"]:
        f = rng.uniform(5.5, 9.0)
        amp = 0.8 * rng.lognormal(0, 0.35)
    else:
        f = rng.uniform(8.0, 12.0)
        amp = 0.25 * rng.lognormal(0, 0.35)
    return f, amp


# --------------------------------------------------------------------------- sensor
def gen_sensor_subject(subj, seconds, rng):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    f, amp = tremor_params(subj, rng)
    # tremor waxes and wanes, frequency drifts slightly
    env = 1 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f + 0.15 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))) / FS
    brady = 1 - 0.12 * subj["severity"] if subj["label"] == 1 else 1.0
    dirs = rng.normal(size=(2, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    trem = amp * 0.05 * env * np.sin(phase)                 # in g
    trem2 = 0.5 * amp * 0.05 * env * np.sin(2 * phase + 0.7)  # harmonic
    # voluntary low-frequency hand movement
    vol = np.zeros((n, 3))
    for _ in range(4):
        vol += brady * rng.uniform(0.05, 0.25) * np.sin(
            2 * np.pi * rng.uniform(0.1, 2.0) * t[:, None] + rng.uniform(0, 6.28, 3))
    g = rng.normal(size=3)
    g /= np.linalg.norm(g)
    acc = g + vol + np.outer(trem, dirs[0]) + np.outer(trem2, dirs[1])
    acc += rng.normal(0, 0.03 * (1 + SHIFT), (n, 3)) + np.cumsum(rng.normal(0, 2e-4, (n, 3)), axis=0)
    # gyro (deg/s): rotational component of tremor + voluntary rotation
    gdir = rng.normal(size=3)
    gdir /= np.linalg.norm(gdir)
    gyr = np.outer(amp * 6 * env * np.cos(phase), gdir) + 20 * np.gradient(vol, axis=0) * FS / 10
    gyr *= 1 - 0.25 * SHIFT                               # different gyro gain
    gyr += rng.normal(0, 1.5 * (1 + SHIFT), (n, 3))
    # magnetometer (uT): orientation dependent, uninformative
    mag = 45 * np.tile(rng.normal(size=3) / 1.7, (n, 1)) + rng.normal(0, 0.8, (n, 3))
    df = pd.DataFrame(np.c_[t, acc, gyr, mag],
                      columns=["t", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"])
    return df.round(5)


# --------------------------------------------------------------------------- spiral
def gen_spiral_trajectory(subj, rng):
    """Pen trajectory (x, y, pressure) of an Archimedean spiral, 3 turns."""
    f, amp = tremor_params(subj, rng)
    s = subj["severity"] if subj["label"] == 1 else 0
    duration = rng.uniform(6, 9) * (1 + 0.15 * s)  # bradykinesia -> slower
    n = int(duration * FS)
    t = np.arange(n) / FS
    theta = 6 * np.pi * np.linspace(0, 1, n) ** rng.uniform(0.9, 1.1)  # non-uniform drawing speed
    micro = 1 - 0.04 * s * theta / theta.max()             # micrographia: shrinks while drawing
    r = 0.9 * theta / theta.max() * micro * rng.uniform(0.85, 1.0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # tremor displacement, amplitude modulated
    env = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
    a = 0.012 * amp * env
    x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    # slow drawing wobble present in everyone
    x += 0.015 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t)
    y += 0.015 * np.cos(2 * np.pi * rng.uniform(0.2, 0.6) * t)
    x += rng.normal(0, 0.002, n)
    y += rng.normal(0, 0.002, n)
    base_p = rng.uniform(0.6, 1.0) * (1 - 0.08 * s) * (1 - 0.3 * SHIFT)  # lighter pen
    p = np.clip(base_p + (0.05 + 0.03 * s) * rng.normal(size=n).cumsum() / np.sqrt(n), 0.2, 1.0)
    return np.c_[x, y, p]


def render_trajectory(traj, size=64, canvas=256):
    """Render (x, y, pressure) in [-1, 1] to a grayscale image, ink = dark."""
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    xy = (traj[:, :2] * 0.48 + 0.5) * canvas
    for i in range(len(xy) - 1):
        w = max(1, int(round(1 + 3 * traj[i, 2] + 2 * SHIFT)))  # thicker nib
        d.line([tuple(xy[i]), tuple(xy[i + 1])], fill=int(90 * (1 - traj[i, 2])), width=w)
    return img.resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------- video
def gen_video_clip(subj, rng, hand, T=16, H=64, fps=15):
    f, amp = tremor_params(subj, rng)
    t = np.arange(T) / fps
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
    bg = hand["bg"] + rng.normal(0, 4, (H, H))
    ph = rng.uniform(0, 6.28)
    px = 1.1 * amp * np.sin(2 * np.pi * f * t + ph)                  # translational tremor (px)
    py = 0.8 * amp * np.sin(2 * np.pi * f * t + ph + 1.2)
    rot = 0.04 * amp * np.sin(2 * np.pi * f * t + ph + 0.5)          # pronation/supination
    drift = rng.normal(0, 0.6, (T, 2)).cumsum(0) * 0.5                 # voluntary + camera shake
    frames = np.empty((T, H, H), np.float32)
    for k in range(T):
        cx = hand["cx"] + px[k] + drift[k, 0]
        cy = hand["cy"] + py[k] + drift[k, 1]
        a = hand["angle"] + rot[k]
        ca, sa = np.cos(a), np.sin(a)
        u = (xx - cx) * ca + (yy - cy) * sa
        v = -(xx - cx) * sa + (yy - cy) * ca
        mask = (u / hand["pw"]) ** 2 + (v / hand["ph"]) ** 2 < 1                    # palm
        for j, off in enumerate(np.linspace(-0.7, 0.7, 4)):                       # fingers
            fu = u - off * hand["pw"]
            mask |= (np.abs(fu) < hand["fw"]) & (v < -0.6 * hand["ph"]) & (v > -hand["ph"] - hand["fl"][j])
        mask |= (u > 0.8 * hand["pw"]) & (u < 0.8 * hand["pw"] + hand["fl"][0] * 0.8) & (np.abs(v) < hand["fw"])  # thumb
        img = bg.copy()
        img[mask] = hand["skin"]
        frames[k] = img
    frames *= rng.uniform(0.85, 1.15) * (1 - 0.3 * SHIFT)        # darker camera
    frames += rng.normal(0, 6 * (1 + SHIFT), frames.shape)
    return np.clip(frames, 0, 255).astype(np.uint8)


def random_hand(rng, H=64):
    return dict(cx=H / 2 + rng.normal(0, 3), cy=H / 2 + 6 + rng.normal(0, 3),
                angle=rng.normal(0, 0.25), pw=rng.uniform(9, 12) * (1 - 0.2 * SHIFT), ph=rng.uniform(10, 13) * (1 - 0.2 * SHIFT),
                fw=rng.uniform(1.6, 2.3), fl=rng.uniform(8, 13, 4),
                skin=rng.uniform(150, 220), bg=rng.uniform(30, 110))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=0.0, help="domain shift for an external cohort (e.g. 1.0)")
    ap.add_argument("--sensor-subjects", type=int, default=30, help="per class")
    ap.add_argument("--sensor-seconds", type=float, default=30.0)
    ap.add_argument("--spiral-subjects", type=int, default=40, help="per class")
    ap.add_argument("--spirals-per-subject", type=int, default=4)
    ap.add_argument("--video-subjects", type=int, default=40, help="per class")
    ap.add_argument("--clips-per-subject", type=int, default=8)
    a = ap.parse_args()
    global SHIFT
    SHIFT = a.shift
    rng = np.random.default_rng(a.seed)
    meta = []

    # sensor (MPU-9250-like)
    d = os.path.join(a.out, "sensor")
    os.makedirs(d, exist_ok=True)
    for s in make_subjects(a.sensor_subjects, a.sensor_subjects, "S", rng):
        gen_sensor_subject(s, a.sensor_seconds, rng).to_csv(
            os.path.join(d, f"{s['subject']}.csv.gz"), index=False)
        meta.append(dict(modality="sensor", **s))
    print("sensor done")

    # spiral drawings (Dryad / Kaggle-like image folders)
    for s in make_subjects(a.spiral_subjects, a.spiral_subjects, "D", rng):
        cls = "parkinson" if s["label"] else "healthy"
        d = os.path.join(a.out, "spiral", cls)
        os.makedirs(d, exist_ok=True)
        for k in range(a.spirals_per_subject):
            render_trajectory(gen_spiral_trajectory(s, rng)).save(os.path.join(d, f"{s['subject']}_{k}.png"))
        meta.append(dict(modality="spiral", **s))
    print("spiral done")

    # hand-motion videos
    clips, labels, subjects = [], [], []
    for s in make_subjects(a.video_subjects, a.video_subjects, "V", rng):
        hand = random_hand(rng)
        for _ in range(a.clips_per_subject):
            clips.append(gen_video_clip(s, rng, hand))
            labels.append(s["label"])
            subjects.append(s["subject"])
        meta.append(dict(modality="video", **s))
    d = os.path.join(a.out, "video")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "clips.npz"), X=np.stack(clips),
                        y=np.array(labels), subject=np.array(subjects))
    print("video done")

    pd.DataFrame(meta).to_csv(os.path.join(a.out, "subjects.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
