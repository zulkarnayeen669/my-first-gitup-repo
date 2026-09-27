"""
End-to-end replication of
  G. Chandra, T. K. Gandhi, B. Singh, "A Hybrid Multimodal Deep Learning
  Framework for Robust Diagnosis of Parkinson's Disease Using Hand Tremor
  Analysis", IEEE Sensors Journal, 26(3):4723-4730, 2026.

Pipeline (Fig. 3 / Algorithm 1):
  Phase 1  load + normalise the three modalities, subject-level hold-out split
  Phase 2  train the unimodal subnetworks independently (LSTM, 2-D CNN, 3-D CNN)
           -> Table IV, Fig. 6(e)(f)(g)
  Phase 3  late feature-level fusion: Concat(Z_sensor, Z_spiral, Z_video) in R^160
  Phase 4+ hybrid GRU-LSTNet + BAM head, trained end-to-end
           (Adam lr=1e-3, 10 epochs, batch 8, val split 0.2, dropout 0.5, CE loss)
           -> Table II, Fig. 5, Fig. 6(b)
  Baselines  SVM (RBF), Random Forest, k-NN (k=5), Logistic Regression -> Table III, Fig. 6
  Table V  inference time per sample and parameter count

Usage:
  python data/generate_synthetic.py            # or put the real datasets under data/real
  python run_replication.py --data-root data/synthetic --out results
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from datasets import load_sensor, load_spiral, load_video, pair_across_modalities, subject_split  # noqa: E402
from features import sensor_features, spiral_features, video_features  # noqa: E402
from models import (MultimodalNet, SensorLSTM, Spiral2DCNN,  # noqa: E402
                    UnimodalClassifier, Video3DCNN, n_params)

MODS = ["sensor", "spiral", "video"]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, prob),
                tnr=cm[0, 0] / cm[0].sum(), tpr=cm[1, 1] / cm[1].sum())


# --------------------------------------------------------------------------- training loop
def fit(model, make_batch, n_train, n_val, epochs, bs, lr, log):
    """Adam + categorical cross-entropy (Eq. 8); keeps the weights of the epoch
    with the best validation accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        tot = 0.0
        for i in range(0, n_train, bs):
            xb, yb = make_batch("train", perm[i:i + bs])
            opt.zero_grad()
            loss = loss_fn(model(*xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
        prob, yv = predict(model, make_batch, "val", n_val)
        acc = ((prob >= 0.5) == yv).mean()
        log(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / n_train:.4f}  val_acc {acc:.3f}")
        if acc >= best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, make_batch, split, n, bs=64):
    model.eval()
    probs, ys = [], []
    for i in range(0, n, bs):
        xb, yb = make_batch(split, np.arange(i, min(n, i + bs)))
        probs.append(torch.softmax(model(*xb), 1)[:, 1].numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


@torch.no_grad()
def inference_ms(model, inputs, reps=50):
    model.eval()
    model(*inputs)
    t = time.perf_counter()
    for _ in range(reps):
        model(*inputs)
    return (time.perf_counter() - t) / reps * 1e3


# --------------------------------------------------------------------------- figures
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#ecebe6", lw=0.6)
    ax.set_axisbelow(True)


def plot_fig5(m, path):
    import matplotlib.pyplot as plt
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    _style(ax)
    ax.grid(axis="x", visible=False)
    bars = ax.bar(names, [v * 100 for v in vals], width=0.45, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v * 100:.1f}%", ha="center",
                fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)", color=MUTED, fontsize=9)
    ax.set_title("Fig. 5 - Hybrid GRU-LSTNet + BAM, test-set metrics", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fig6(results, y_true, path):
    """Row-normalised confusion matrix + ROC curve for every model (Fig. 6)."""
    import matplotlib.pyplot as plt
    order = list(results)
    fig, axes = plt.subplots(len(order), 2, figsize=(7.2, 2.9 * len(order)))
    for r, name in enumerate(order):
        prob = results[name]["prob"]
        cm = confusion_matrix(y_true, (prob >= 0.5).astype(int), labels=[0, 1]).astype(float)
        cm /= cm.sum(1, keepdims=True)
        ax = axes[r, 0]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Healthy", "Parkinson"], fontsize=8)
        ax.set_yticks([0, 1], ["Healthy", "Parkinson"], fontsize=8)
        ax.set_xlabel("Predicted label", fontsize=8, color=MUTED)
        ax.set_ylabel("True label", fontsize=8, color=MUTED)
        ax.set_title(f"{name} - confusion matrix", fontsize=9, color=INK, loc="left")
        ax = axes[r, 1]
        _style(ax)
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY)
        ax.plot(fpr, tpr, lw=2, color=BLUE)
        ax.text(0.97, 0.05, f"AUC = {results[name]['metrics']['auc']:.2f}", ha="right", fontsize=9, color=INK)
        ax.set_xlabel("False positive rate", fontsize=8, color=MUTED)
        ax.set_ylabel("True positive rate", fontsize=8, color=MUTED)
        ax.set_title(f"{name} - ROC", fontsize=9, color=INK, loc="left")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)       # Sec. II-H
    ap.add_argument("--batch-size", type=int, default=8)    # Sec. II-H
    ap.add_argument("--lr", type=float, default=1e-3)       # Sec. II-H
    ap.add_argument("--val-split", type=float, default=0.2)  # Sec. II-H
    ap.add_argument("--test-size", type=float, default=0.3, help="fraction of subjects held out")
    ap.add_argument("--n-fused-train", type=int, default=1200)
    ap.add_argument("--n-fused-test", type=int, default=600)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--lstm-readout", choices=["mean", "last"], default="mean",
                    help="'last' = paper's Dense(h_T); 'mean' = temporal mean of LSTM states")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    # ---------------- Phase 1: data preparation
    t0 = time.time()
    data = dict(sensor=load_sensor(os.path.join(a.data_root, "sensor")),
                spiral=load_spiral(os.path.join(a.data_root, "spiral")),
                video=load_video(os.path.join(a.data_root, "video")))
    splits = {}
    for m in MODS:
        d = data[m]
        tr, te = subject_split(d, a.test_size, rng)
        sub = type(d)(d.X[tr], d.y[tr], d.groups[tr])
        tr2, va2 = subject_split(sub, a.val_split, rng)
        splits[m] = dict(train=tr[tr2], val=tr[va2], test=te)
        log(f"[data] {m:6s} X{tuple(d.X.shape)}  subjects={len(np.unique(d.groups))}  "
            f"train/val/test samples = {len(tr[tr2])}/{len(tr[va2])}/{len(te)}")
    ys = [data[m].y for m in MODS]
    fused = {}
    for split, n in (("train", a.n_fused_train), ("val", int(a.n_fused_train * a.val_split)),
                     ("test", a.n_fused_test)):
        idx, lab = pair_across_modalities([splits[m][split] for m in MODS], ys, n, rng)
        fused[split] = (idx, lab)
    y_test = fused["test"][1]
    log(f"[data] fused triples train/val/test = "
        f"{len(fused['train'][1])}/{len(fused['val'][1])}/{len(y_test)}  ({time.time() - t0:.0f}s)")

    T = {m: torch.from_numpy(data[m].X.astype(np.float32)) for m in MODS}
    results = {}

    # ---------------- Phase 2: unimodal subnetworks (trained independently)
    encoders = {}
    for mi, (m, enc, title) in enumerate([("sensor", lambda: SensorLSTM(readout=a.lstm_readout), "LSTM"), ("spiral", Spiral2DCNN, "2D CNN"),
                                          ("video", Video3DCNN, "3D CNN")]):
        log(f"[unimodal] training {title} on the {m} modality")
        set_seed(a.seed + mi)
        model = UnimodalClassifier(enc())
        sp = splits[m]

        def batch_own(split, ix, m=m, sp=sp):
            j = sp[split][ix]
            return (T[m][j],), torch.from_numpy(data[m].y[j]).long()

        t = time.time()
        fit(model, batch_own, len(sp["train"]), len(sp["val"]), a.epochs, a.batch_size, a.lr, log)
        own_prob, own_y = predict(model, batch_own, "test", len(sp["test"]))

        def batch_fused(split, ix, m=m, mi=mi):   # evaluate on the shared fused test triples
            j = fused[split][0][ix, mi]
            return (T[m][j],), torch.from_numpy(fused[split][1][ix]).long()

        prob, _ = predict(model, batch_fused, "test", len(y_test))
        results[title] = dict(prob=prob, metrics=metrics(y_test, prob),
                              own_test_metrics=metrics(own_y, own_prob),
                              params=n_params(model), train_s=time.time() - t,
                              infer_ms=inference_ms(model, (T[m][:1],)))
        log(f"    -> {title}: fused-test {results[title]['metrics']}")
        encoders[m] = model.encoder

    # ---------------- Phases 3-6: fusion + hybrid GRU-LSTNet + BAM, end-to-end
    log("[proposed] training hybrid GRU-LSTNet + BAM end-to-end on fused triples")
    set_seed(a.seed + 10)
    model = MultimodalNet(encoders["sensor"], encoders["spiral"], encoders["video"])

    def batch_mm(split, ix):
        j = fused[split][0][ix]
        return tuple(T[m][j[:, k]] for k, m in enumerate(MODS)), torch.from_numpy(fused[split][1][ix]).long()

    t = time.time()
    fit(model, batch_mm, len(fused["train"][1]), len(fused["val"][1]), a.epochs, a.batch_size, a.lr, log)
    prob, _ = predict(model, batch_mm, "test", len(y_test))
    one = tuple(T[m][:1] for m in MODS)
    results["Proposed Model"] = dict(prob=prob, metrics=metrics(y_test, prob), params=n_params(model),
                                     train_s=time.time() - t, infer_ms=inference_ms(model, one))
    log(f"    -> Proposed: {results['Proposed Model']['metrics']}")

    # simple reference: average the three unimodal probabilities (no learned fusion)
    avg = np.mean([results[k]["prob"] for k in ("LSTM", "2D CNN", "3D CNN")], 0)
    extra = {"Unimodal probability averaging": metrics(y_test, avg)}

    # ---------------- classical ML baselines on handcrafted multimodal features
    log("[baselines] handcrafted features + SVM / RF / k-NN / LR")
    feats = [sensor_features(data["sensor"].X), spiral_features(data["spiral"].X),
             video_features(data["video"].X)]

    def fused_feats(split):
        idx = fused[split][0]
        return np.concatenate([feats[k][idx[:, k]] for k in range(3)], 1)

    Xtr = np.concatenate([fused_feats("train"), fused_feats("val")])
    ytr = np.concatenate([fused["train"][1], fused["val"][1]])
    Xte = fused_feats("test")
    classical = {
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=a.seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=a.seed, n_jobs=-1),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000),
    }
    for name, clf in classical.items():
        pipe = make_pipeline(StandardScaler(), clf)
        t = time.time()
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        t_inf = time.perf_counter()
        for i in range(50):
            pipe.predict(Xte[i:i + 1])
        results[name] = dict(prob=prob, metrics=metrics(y_test, prob), params=None,
                             train_s=time.time() - t, infer_ms=(time.perf_counter() - t_inf) / 50 * 1e3)
        log(f"    -> {name}: {results[name]['metrics']}")

    # ---------------- tables & figures
    pct = lambda v: f"{100 * v:.2f}%"  # noqa: E731

    def table(rows):
        return pd.DataFrame([dict(Model=r, Accuracy=pct(results[r]["metrics"]["accuracy"]),
                                  Precision=pct(results[r]["metrics"]["precision"]),
                                  Recall=pct(results[r]["metrics"]["recall"]),
                                  **{"F1-Score": pct(results[r]["metrics"]["f1"]),
                                     "ROC-AUC": f"{results[r]['metrics']['auc']:.3f}",
                                     "TPR": pct(results[r]["metrics"]["tpr"]),
                                     "TNR": pct(results[r]["metrics"]["tnr"])}) for r in rows])

    pm = results["Proposed Model"]["metrics"]
    t2 = pd.DataFrame(dict(Metric=["Accuracy", "Precision", "Recall", "F1-Score"],
                           Value=[pct(pm[k]) for k in ("accuracy", "precision", "recall", "f1")]))
    t3 = table(["SVM (RBF)", "Random Forest", "k-NN (k=5)", "Logistic Regression", "Proposed Model"])
    t4 = table(["LSTM", "2D CNN", "3D CNN", "Proposed Model"])
    t4.insert(0, "Modality", ["Sensor", "Spiral", "Video", "Fused"])
    kinds = {"SVM (RBF)": "Classical ML", "Random Forest": "Ensemble ML", "k-NN (k=5)": "Instance-based ML",
             "Logistic Regression": "Linear ML", "LSTM": "RNN-based DL", "2D CNN": "Spatial CNN",
             "3D CNN": "Spatio-temporal CNN", "Proposed Model": "GRU-LSTNet + BAM"}
    t5 = pd.DataFrame([dict(Model=k, **{"Inference time (ms)": f"{results[k]['infer_ms']:.2f}",
                                        "Model type": v,
                                        "Params (M)": "-" if results[k]["params"] is None
                                        else f"{results[k]['params'] / 1e6:.2f}"}) for k, v in kinds.items()])
    t_extra = pd.DataFrame([dict(Model=k, **{kk: f"{vv:.4f}" for kk, vv in v.items()}) for k, v in extra.items()]
                           + [dict(Model=f"{k} (own test split)", **{kk: f"{vv:.4f}" for kk, vv in
                                                                    results[k]["own_test_metrics"].items()})
                              for k in ("LSTM", "2D CNN", "3D CNN")])
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        for title, df in [("Table II - Performance metrics of the proposed model", t2),
                          ("Table III - Comparison with machine learning models", t3),
                          ("Table IV - Comparison across modalities", t4),
                          ("Table V - Inference time per sample (CPU, batch 1) and model complexity", t5),
                          ("Extra - probability-averaging reference and unimodal scores on their own test splits",
                           t_extra)]:
            f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n\n")
            df.to_csv(os.path.join(a.out, title.split(" - ")[0].replace(" ", "_").lower() + ".csv"), index=False)
    plot_fig5(pm, os.path.join(a.out, "fig5_metrics.png"))
    order = ["SVM (RBF)", "Proposed Model", "Random Forest", "k-NN (k=5)", "LSTM", "2D CNN", "3D CNN",
             "Logistic Regression"]   # panel order (a)-(h) of Fig. 6
    plot_fig6({k: results[k] for k in order}, y_test, os.path.join(a.out, "fig6_confusion_roc.png"))
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "prob"} for k, v in results.items()} | {"extra": extra},
              open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    log("\n" + open(os.path.join(a.out, "tables.md")).read())
    log(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
