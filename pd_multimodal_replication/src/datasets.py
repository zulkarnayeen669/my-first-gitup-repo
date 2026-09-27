"""
Data loading, preprocessing ("Phase 1: Data Preparation", Algorithm 1) and the
cross-cohort pairing used for late feature-level fusion (Section II-F).

Each loader returns a `Modality` holding X (samples), y (0 = healthy, 1 = PD)
and `groups` (subject id of every sample) so train/test splits are made at the
subject level -- a window/clip of a test subject is never seen in training.

Real-data layouts accepted (point --data-root at them, see README):
  sensor/   one CSV (or .csv.gz) per recording; columns ax,ay,az,gx,gy,gz
            (+ optional others). Label from a `label` column, from a
            subjects.csv (subject,label), or from file names containing
            "PD"/"parkinson" vs "HC"/"healthy"/"control".
  spiral/   {healthy,parkinson}/**/*.png|jpg   (Kaggle "parkinsons-drawings"
            layout) or UCI/Dryad tablet .txt files "X;Y;Z;Pressure;GripAngle;
            Timestamp;TestID" under {control,parkinson}/ that are rendered to
            images with `render_trajectory`.
  video/    clips.npz (X uint8 N x T x H x W, y, subject)  or
            {healthy,parkinson}/**/*.mp4|avi  (needs opencv-python-headless).
"""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]  # X_t in R^6 (Section II-C)


@dataclass
class Modality:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def _label_from_name(name):
    n = name.lower()
    if re.search(r"(parkinson|^pd|[_/-]pd|pd\d)", n):
        return 1
    if re.search(r"(healthy|control|^hc|[_/-]hc|hc\d)", n):
        return 0
    return None


# --------------------------------------------------------------------------- sensor
def load_sensor(root, win=256, step=128, fs=100.0, hp=3.0, decim=2):
    """Preprocessing ("normalisation, resizing, cleaning", Fig. 3):
    4th-order Butterworth high-pass at `hp` Hz removes gravity, drift and
    voluntary movement; the signal is decimated by `decim` (100 -> 50 Hz) and
    windowed into (win/decim x 6) segments with 50% overlap. Channels are
    scaled by one global per-channel std so tremor *amplitude* is preserved
    (per-recording z-scoring would erase it)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hp, "highpass", fs=fs, output="sos")
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv*"), recursive=True))
    labels_csv = os.path.join(root, "..", "subjects.csv")
    table = {}
    if os.path.exists(labels_csv):
        m = pd.read_csv(labels_csv)
        table = dict(zip(m.subject.astype(str), m.label.astype(int)))
    X, y, g = [], [], []
    for f in files:
        subj = re.sub(r"\.csv(\.gz)?$", "", os.path.basename(f))
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        if "label" in df.columns:
            lab = int(df["label"].iloc[0])
        elif subj in table:
            lab = table[subj]
        else:
            lab = _label_from_name(os.path.relpath(f, root))
        if lab is None:
            continue
        sig = sosfiltfilt(sos, df[SENSOR_COLS].to_numpy(np.float64), axis=0)
        for s in range(0, len(sig) - win + 1, step):
            X.append(sig[s:s + win:decim])
            y.append(lab)
            g.append(subj)
    X = np.stack(X).astype(np.float32)
    X /= X.std(axis=(0, 1), keepdims=True) + 1e-8
    return Modality(X, np.array(y), np.array(g))


# --------------------------------------------------------------------------- spiral
def render_trajectory(traj, size=64, canvas=256):
    """(x, y, pressure) -> grayscale image. Coordinates are min-max scaled."""
    from PIL import ImageDraw
    xy = traj[:, :2].astype(np.float64)
    xy = (xy - xy.min(0)) / (np.ptp(xy, 0).max() + 1e-9)
    xy = (xy * 0.92 + 0.04) * canvas
    p = traj[:, 2] / (traj[:, 2].max() + 1e-9) if traj.shape[1] > 2 else np.ones(len(xy))
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    for i in range(len(xy) - 1):
        d.line([tuple(xy[i]), tuple(xy[i + 1])], fill=int(90 * (1 - p[i])),
               width=max(1, int(round(1 + 3 * p[i]))))
    return img.resize((size, size), Image.LANCZOS)


def load_spiral(root, size=64):
    """Grayscale 64x64x1 images scaled to [0, 1] with ink = 1."""
    X, y, g = [], [], []
    for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".txt"):
            continue
        lab = _label_from_name(os.path.relpath(f, root))
        if lab is None:
            continue
        if ext == ".txt":  # UCI / Dryad tablet trajectory: X;Y;Z;Pressure;GripAngle;Timestamp;TestID
            arr = pd.read_csv(f, sep=";", header=None).to_numpy(np.float64)
            arr = arr[arr[:, 6] == 0] if arr.shape[1] > 6 and (arr[:, 6] == 0).any() else arr  # static spiral test
            img = render_trajectory(arr[:, [0, 1, 3]], size)
        else:
            img = Image.open(f).convert("L").resize((size, size), Image.LANCZOS)
        X.append(1.0 - np.asarray(img, np.float32) / 255.0)
        y.append(lab)
        base = os.path.splitext(os.path.basename(f))[0]
        g.append(re.sub(r"_\d+$", "", base))  # "<subject>_<k>" -> subject
    return Modality(np.stack(X)[:, None], np.array(y), np.array(g))


# --------------------------------------------------------------------------- video
def _read_video(path, T=16, size=64):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (size, size)))
    cap.release()
    idx = np.linspace(0, len(frames) - 1, T).round().astype(int)  # uniform temporal sampling
    return np.stack([frames[i] for i in idx])


def load_video(root, T=16, size=64):
    """Clips of T=16 grayscale 64x64 frames, per-clip standardised."""
    npz = os.path.join(root, "clips.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        X, y, g = d["X"], d["y"], d["subject"]
    else:
        X, y, g = [], [], []
        for f in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
            if os.path.splitext(f)[1].lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                continue
            lab = _label_from_name(os.path.relpath(f, root))
            if lab is None:
                continue
            X.append(_read_video(f, T, size))
            y.append(lab)
            g.append(os.path.splitext(os.path.basename(f))[0].split("_")[0])
        X, y, g = np.stack(X), np.array(y), np.array(g)
    X = X.astype(np.float32)
    X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / (X.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
    return Modality(X[:, None], np.asarray(y), np.asarray(g))  # N x 1 x T x H x W (PyTorch layout)


# --------------------------------------------------------------------------- splits
def subject_split(m, test_size, rng):
    """Stratified subject-level hold-out split."""
    tr, te = [], []
    for lab in (0, 1):
        subs = np.unique(m.groups[m.y == lab])
        rng.shuffle(subs)
        k = max(1, int(round(test_size * len(subs))))
        te += list(subs[:k])
        tr += list(subs[k:])
    tr_idx = np.where(np.isin(m.groups, tr))[0]
    te_idx = np.where(np.isin(m.groups, te))[0]
    return tr_idx, te_idx


def pair_across_modalities(idx_by_mod, ys, n, rng):
    """Late feature-level fusion across *different cohorts* (Section II-F):
    the three datasets contain different people, so a fused sample is a
    triple (sensor_i, spiral_j, video_k) drawn from the same class. Returns
    an (n, 3) index array and labels; classes are balanced."""
    out, lab = [], []
    for c in (0, 1):
        pools = [idx[ys[m][idx] == c] for m, idx in enumerate(idx_by_mod)]
        k = n // 2
        cols = [rng.choice(p, k, replace=len(p) < k) for p in pools]
        out.append(np.stack(cols, 1))
        lab += [c] * k
    out = np.concatenate(out)
    lab = np.array(lab)
    perm = rng.permutation(len(lab))
    return out[perm], lab[perm]
