"""
Extension of the replication with the ideas of
  S. Gujjeti et al., "NeuroCrossAttention fusion for multimodal explainable
  early diagnosis of neurodegenerative diseases", Discover Computing 29:329, 2026.

  1. Cross-modal attention fusion (NCAF Eqs. 3-5) + modality masking      -> src/ncaf.py
  2. Ablation over fusion strategies, 5 seeds, identical frozen encoders
     (unimodal / late fusion / concat / paper's GRU-LSTNet+BAM / cross-attention +- masking)
  3. Robustness: missing modality at test time; one modality corrupted by noise
  4. Explainability (NCAF Sec. 4.4, Algorithm 3)                          -> src/xai.py
       global cross-modal attention, gradient saliency (Eq. 6), hybrid map
       E = lambda*A + (1-lambda)*S (Eq. 7) with a deletion faithfulness test,
       Grad-CAM for spirals (2-D) and videos (3-D) with localisation scores,
       frequency occlusion + time saliency for the sensor, SHAP for the
       handcrafted-feature model
  5. Cross-cohort generalisation on a domain-shifted external cohort:
       direct transfer, unsupervised domain adaptation (embedding alignment),
       transfer learning (fine-tune the fusion head on 30% of external subjects)

Fusion heads are trained on embeddings from the frozen, pre-trained unimodal
encoders of the main replication (same subjects, splits and seeds), so every
head sees exactly the same information. Protocol as in Chandra et al.:
Adam lr 1e-3, 10 epochs, batch 8, CE loss, best-validation-epoch weights.

Usage:
  python run_extension.py --data-root data/synthetic --external-root data/synthetic_external
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from datasets import load_sensor, load_spiral, load_video, pair_across_modalities, subject_split  # noqa: E402
from features import feature_names, sensor_features, spiral_features, video_features  # noqa: E402
from models import HybridGRULSTNetBAM, SensorLSTM, Spiral2DCNN, UnimodalClassifier, Video3DCNN  # noqa: E402
from ncaf import ConcatMLPHead, CrossModalAttentionFusion, EncodersPlusHead, HeadOnEmbeddings  # noqa: E402
import xai  # noqa: E402
from run_replication import MODS, fit, metrics, predict, prepare_data, set_seed, _style  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
MOD_COLORS = dict(sensor=BLUE, spiral=ORANGE, video=AQUA)
MOD_TITLES = dict(sensor="Sensor", spiral="Spiral", video="Video")
BANDS = [(3, 5), (5, 7), (7, 9), (9, 12), (12, 25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--external-root", default="data/synthetic_external")
    ap.add_argument("--out", default="results_extension")
    ap.add_argument("--ckpt-dir", default="results/checkpoints", help="reuse unimodal encoders if present")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--n-fused-train", type=int, default=1200)
    ap.add_argument("--n-fused-test", type=int, default=600)
    ap.add_argument("--p-mask", type=float, default=0.2, help="modality-masking probability")
    ap.add_argument("--lam", type=float, default=0.5, help="lambda in E = lambda*A + (1-lambda)*S")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    t0 = time.time()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data, splits, fused = prepare_data(a.data_root, rng, a.test_size, a.val_split,
                                       a.n_fused_train, a.n_fused_test, log)
    y_test = fused["test"][1]
    T = {m: torch.from_numpy(data[m].X.astype(np.float32)) for m in MODS}

    # ------------------------------------------------------------------ 1. frozen unimodal encoders
    encoders, uni_prob = {}, {}
    for mi, (m, enc, title) in enumerate([("sensor", SensorLSTM, "LSTM"), ("spiral", Spiral2DCNN, "2D CNN"),
                                          ("video", Video3DCNN, "3D CNN")]):
        model = UnimodalClassifier(enc())
        paths = [os.path.join(d, f"unimodal_{m}.pt") for d in (a.ckpt_dir, os.path.join(a.out, "checkpoints"))]
        found = next((p for p in paths if os.path.exists(p)), None)
        sp = splits[m]

        def batch_own(split, ix, m=m, sp=sp):
            j = sp[split][ix]
            return (T[m][j],), torch.from_numpy(data[m].y[j]).long()

        if found:
            model.load_state_dict(torch.load(found))
            log(f"[encoders] loaded {title} from {found}")
        else:  # same protocol and seed as run_replication.py
            log(f"[encoders] training {title} (no checkpoint found)")
            set_seed(a.seed + mi)
            fit(model, batch_own, len(sp["train"]), len(sp["val"]), a.epochs, a.batch_size, a.lr, log)
            torch.save(model.state_dict(), paths[1])

        def batch_fused(split, ix, m=m, mi=mi):
            j = fused[split][0][ix, mi]
            return (T[m][j],), torch.from_numpy(fused[split][1][ix]).long()

        uni_prob[title] = predict(model, batch_fused, "test", len(y_test))[0]
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        encoders[m] = model

    @torch.no_grad()
    def embed(m, X, bs=64):
        return torch.cat([encoders[m].encoder(X[i:i + bs]) for i in range(0, len(X), bs)])

    Z = {m: embed(m, T[m]) for m in MODS}
    log(f"[encoders] embeddings ready ({time.time() - t0:.0f}s)")

    def batch_z(fz, Zd):
        def make(split, ix):
            j = fz[split][0][ix]
            return ([Zd[m][j[:, k]] for k, m in enumerate(MODS)],), torch.from_numpy(fz[split][1][ix]).long()
        return make

    make_int = batch_z(fused, Z)

    # ------------------------------------------------------------------ 2. ablation over fusion heads
    variants = {
        "Concat + MLP": lambda: ConcatMLPHead(),
        "Concat + MLP + masking": lambda: ConcatMLPHead(p_mask=a.p_mask),
        "GRU-LSTNet + BAM (paper head)": lambda: HeadOnEmbeddings(HybridGRULSTNetBAM()),
        "Cross-attention (NCAF)": lambda: CrossModalAttentionFusion(),
        "Cross-attention + masking": lambda: CrossModalAttentionFusion(p_mask=a.p_mask),
    }
    heads, abl = {}, {}
    quiet = lambda s: None  # noqa: E731
    for name, ctor in variants.items():
        runs = []
        for k in range(a.n_seeds):
            set_seed(a.seed + 100 + k)
            h = ctor()
            fit(h, make_int, len(fused["train"][1]), len(fused["val"][1]), a.epochs, a.batch_size, a.lr, quiet)
            prob, _ = predict(h, make_int, "test", len(y_test))
            runs.append(metrics(y_test, prob))
            if k == 0:
                heads[name] = h
        abl[name] = runs
        log(f"[ablation] {name:32s} acc {np.mean([r['accuracy'] for r in runs]):.4f} "
            f"+- {np.std([r['accuracy'] for r in runs]):.4f}  auc {np.mean([r['auc'] for r in runs]):.4f}")
    late = np.mean([uni_prob[t] for t in ("LSTM", "2D CNN", "3D CNN")], 0)
    single = {f"{t} only": [metrics(y_test, uni_prob[t])] for t in ("LSTM", "2D CNN", "3D CNN")}
    abl = {**single, "Late fusion (probability averaging)": [metrics(y_test, late)], **abl}
    torch.save(heads["Cross-attention + masking"].state_dict(), os.path.join(a.out, "checkpoints", "ncaf_masking.pt"))
    main_json = os.path.join(os.path.dirname(a.ckpt_dir), "results.json")
    e2e = json.load(open(main_json)).get("Proposed Model", {}).get("metrics") if os.path.exists(main_json) else None

    # ------------------------------------------------------------------ 3a. missing modality
    conds = {"all": [1, 1, 1], "no sensor": [0, 1, 1], "no spiral": [1, 0, 1], "no video": [1, 1, 0]}

    @torch.no_grad()
    def eval_head(h, Zd, fz, present=None, split="test"):
        h.eval()
        idx, lab = fz[split]
        zs = [Zd[m][idx[:, k]] for k, m in enumerate(MODS)]
        pr = None if present is None else torch.tensor(present, dtype=torch.bool).expand(len(lab), 3)
        return torch.softmax(h(zs, pr), 1)[:, 1].numpy(), lab

    miss = {}
    for name, h in heads.items():
        miss[name] = {c: metrics(*eval_head(h, Z, fused, v)[::-1])["accuracy"] for c, v in conds.items()}
    uni_list = [uni_prob["LSTM"], uni_prob["2D CNN"], uni_prob["3D CNN"]]
    miss["Late fusion (probability averaging)"] = {
        c: metrics(y_test, np.mean([p for p, keep in zip(uni_list, v) if keep], 0))["accuracy"] for c, v in conds.items()}
    log("[missing modality] " + json.dumps(miss, indent=1))

    # ------------------------------------------------------------------ 3b. noisy modality
    g = torch.Generator().manual_seed(a.seed)
    noisy = {}
    ncaf = heads["Cross-attention + masking"]
    for k, m in enumerate(MODS):
        Xn = T[m] + 1.5 * torch.randn(T[m].shape, generator=g) * T[m].std()
        Zn = dict(Z)
        Zn[m] = embed(m, Xn)
        row = {}
        for name, h in heads.items():
            row[name] = metrics(*eval_head(h, Zn, fused)[::-1])["accuracy"]
        for tag, Zd in (("clean", Z), ("noisy", Zn)):
            eval_head(ncaf, Zd, fused)
            row[f"NCAF attention to {m} ({tag})"] = float(ncaf.last_alpha[:, :, k].mean())
        noisy[m] = row
    log("[noisy modality] " + json.dumps(noisy, indent=1))

    # ------------------------------------------------------------------ 4. explainability
    full = EncodersPlusHead(encoders["sensor"].encoder, encoders["spiral"].encoder, encoders["video"].encoder, ncaf)
    for p in full.parameters():  # Grad-CAM needs gradients through the (frozen) encoders
        p.requires_grad_(True)
    idx, lab = fused["test"]
    xs, xi, xv = T["sensor"][idx[:, 0]], T["spiral"][idx[:, 1]], T["video"][idx[:, 2]]
    alpha = xai.attention_matrix(full, xs, xi, xv)
    sal = xai.embedding_saliency(full, xs, xi, xv)
    E, A, S = xai.hybrid_modality_scores(alpha, sal, a.lam)
    rs = np.random.default_rng(a.seed).random(E.shape)
    faith = {}
    for tag, sc in (("attention only (lambda=1)", A), (f"hybrid (lambda={a.lam})", E),
                    ("saliency only (lambda=0)", S), ("random ranking", rs)):
        top, bot = xai.modality_deletion(full, xs, xi, xv, sc)
        faith[tag] = dict(delta_p_remove_top=top, delta_p_remove_bottom=bot)
    xai_summary = dict(
        attention_mean=alpha.mean(0).tolist(),
        attention_mean_pd=alpha[lab == 1].mean(0).tolist(), attention_mean_hc=alpha[lab == 0].mean(0).tolist(),
        modality_importance=dict(attention=A.mean(0).tolist(), saliency=S.mean(0).tolist(), hybrid=E.mean(0).tolist()),
        top_modality_agreement_attention_vs_saliency=float((A.argmax(1) == S.argmax(1)).mean()),
        faithfulness=faith)
    log("[xai] " + json.dumps(xai_summary, indent=1))

    # Grad-CAM, spiral: ink (dilated) is the region of interest
    n_cam = 240
    cam_s, p_s = xai.gradcam_2d(full, xs[:n_cam], xi[:n_cam], xv[:n_cam])
    ink = torch.nn.functional.max_pool2d((xi[:n_cam] > 0.1).float(), 5, 1, 2)[:, 0].numpy()
    in_s, area_s = xai.cam_localisation(cam_s, ink)
    # Grad-CAM, video: moving-hand region = top 15% temporal-std pixels of the clip
    cams_v, p_v = [], []
    for i in range(0, 96, 16):
        c, p = xai.gradcam_3d(full, xs[i:i + 16], xi[i:i + 16], xv[i:i + 16])
        cams_v.append(c)
        p_v.append(p)
    cam_v, p_v = np.concatenate(cams_v), np.concatenate(p_v)
    tstd = xv[:96, 0].std(1).numpy()
    motion = tstd > np.quantile(tstd.reshape(96, -1), 0.85, axis=1)[:, None, None]
    in_v, area_v = xai.cam_localisation(cam_v.mean(1), motion)
    xai_summary["gradcam_spiral"] = dict(share_on_ink=float(in_s.mean()), ink_area_share=float(area_s.mean()),
                                         ratio=float(in_s.mean() / area_s.mean()))
    xai_summary["gradcam_video"] = dict(share_on_moving_hand=float(in_v.mean()), region_area_share=float(area_v.mean()),
                                        ratio=float(in_v.mean() / area_v.mean()))
    # sensor: frequency occlusion in sensor-only mode (possible thanks to modality masking) and full mode
    only_sensor = torch.tensor([True, False, False]).expand(len(lab), 3)
    occ_s, _ = xai.frequency_occlusion(full, xs, xi, xv, BANDS, present=only_sensor)
    occ_f, _ = xai.frequency_occlusion(full, xs, xi, xv, BANDS)
    xai_summary["sensor_frequency_occlusion"] = {
        mode: {cls: {f"{b0}-{b1} Hz": float(occ[lab == c, k].mean()) for k, (b0, b1) in enumerate(BANDS)}
               for cls, c in (("PD", 1), ("healthy", 0))}
        for mode, occ in (("sensor only", occ_s), ("all modalities", occ_f))}
    tsal = xai.time_saliency(full, xs[:64], xi[:64], xv[:64], only_sensor[:64])
    log("[xai] grad-cam + frequency: " + json.dumps({k: xai_summary[k] for k in
                                                     ("gradcam_spiral", "gradcam_video", "sensor_frequency_occlusion")},
                                                    indent=1))

    # SHAP on the handcrafted-feature random forest (the tabular model)
    import shap
    from sklearn.ensemble import RandomForestClassifier
    feats = [sensor_features(data["sensor"].X), spiral_features(data["spiral"].X), video_features(data["video"].X)]

    def ff(fz, split, fe=feats):
        i = fz[split][0]
        return np.concatenate([fe[k][i[:, k]] for k in range(3)], 1)

    rf = RandomForestClassifier(n_estimators=200, random_state=a.seed, n_jobs=-1)
    rf.fit(np.concatenate([ff(fused, "train"), ff(fused, "val")]),
           np.concatenate([fused["train"][1], fused["val"][1]]))
    Xte_f = ff(fused, "test")[:300]
    sv = shap.TreeExplainer(rf).shap_values(Xte_f)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    names = np.array(feature_names())
    mean_abs = np.abs(sv).mean(0)
    top = np.argsort(mean_abs)[::-1][:15]
    groups = np.array([n.split()[0] for n in names])
    xai_summary["shap_top_features"] = [(names[i], float(mean_abs[i])) for i in top]
    xai_summary["shap_modality_share"] = {m: float(mean_abs[groups == m].sum() / mean_abs.sum()) for m in MODS}
    log("[shap] " + json.dumps({k: xai_summary[k] for k in ("shap_top_features", "shap_modality_share")}, indent=1))

    # ------------------------------------------------------------------ 5. cross-cohort generalisation
    ext = dict(sensor=load_sensor(os.path.join(a.external_root, "sensor")),
               spiral=load_spiral(os.path.join(a.external_root, "spiral")),
               video=load_video(os.path.join(a.external_root, "video")))
    ext_rng = np.random.default_rng(a.seed + 7)
    ext_split = {}
    for m in MODS:
        adapt, test = subject_split(ext[m], 0.7, ext_rng)  # 30% of subjects labelled for adaptation
        ext_split[m] = dict(adapt=adapt, test=test)
    ys_e = [ext[m].y for m in MODS]
    fused_e = {s: pair_across_modalities([ext_split[m][s] for m in MODS], ys_e, n, ext_rng)
               for s, n in (("adapt", 300), ("test", 400))}
    Ze = {m: embed(m, torch.from_numpy(ext[m].X.astype(np.float32))) for m in MODS}
    # unsupervised domain adaptation: align each embedding dimension's mean/std to the source
    # (uses unlabelled external samples only)
    Za = {m: (Ze[m] - Ze[m].mean(0)) / (Ze[m].std(0) + 1e-6) * Z[m][splits[m]["train"]].std(0)
          + Z[m][splits[m]["train"]].mean(0) for m in MODS}
    cross = {}
    for name in ("Concat + MLP", "GRU-LSTNet + BAM (paper head)", "Cross-attention + masking"):
        h = heads[name]
        row = {"internal test": metrics(*eval_head(h, Z, fused)[::-1])}
        row["direct transfer"] = metrics(*eval_head(h, Ze, fused_e)[::-1])
        row["domain adaptation"] = metrics(*eval_head(h, Za, fused_e)[::-1])
        # transfer learning: fine-tune the head on the labelled 30% (aligned embeddings), 10 epochs, lr 1e-4
        ht = copy.deepcopy(h)
        set_seed(a.seed)
        opt = torch.optim.Adam(ht.parameters(), lr=1e-4)
        ia, la = fused_e["adapt"]
        for _ in range(a.epochs):
            ht.train()
            for b in np.array_split(np.random.permutation(len(la)), max(1, len(la) // a.batch_size)):
                zs = [Za[m][ia[b, k]] for k, m in enumerate(MODS)]
                opt.zero_grad()
                nn.functional.cross_entropy(ht(zs), torch.from_numpy(la[b]).long()).backward()
                opt.step()
        row["transfer learning"] = metrics(*eval_head(ht, Za, fused_e)[::-1])
        cross[name] = row
        log(f"[cross-cohort] {name}: " + ", ".join(f"{k} {v['accuracy']:.3f}/{v['auc']:.3f}" for k, v in row.items()))

    # ------------------------------------------------------------------ tables
    def ms(runs, k):
        v = [r[k] for r in runs]
        return f"{100 * np.mean(v):.2f}" + (f" ± {100 * np.std(v):.2f}" if len(v) > 1 else "")

    t_abl = pd.DataFrame([{"Configuration": n, "Seeds": len(r), "Accuracy (%)": ms(r, "accuracy"),
                           "F1 (%)": ms(r, "f1"), "Recall (%)": ms(r, "recall"),
                           "ROC-AUC": f"{np.mean([x['auc'] for x in r]):.3f}"} for n, r in abl.items()])
    if e2e:
        t_abl.loc[len(t_abl)] = {"Configuration": "GRU-LSTNet + BAM, end-to-end (main replication)", "Seeds": 1,
                                 "Accuracy (%)": f"{100 * e2e['accuracy']:.2f}", "F1 (%)": f"{100 * e2e['f1']:.2f}",
                                 "Recall (%)": f"{100 * e2e['recall']:.2f}", "ROC-AUC": f"{e2e['auc']:.3f}"}
    t_miss = pd.DataFrame([{"Model": n, **{c: f"{100 * v:.1f}" for c, v in r.items()}} for n, r in miss.items()])
    t_noise = pd.DataFrame([{"Model": n, **{f"{MOD_TITLES[m]} corrupted": f"{100 * noisy[m][n]:.1f}" for m in MODS}}
                            for n in heads])
    t_att = pd.DataFrame([{"Modality": MOD_TITLES[m],
                           "Attention received, clean": f"{noisy[m][f'NCAF attention to {m} (clean)']:.3f}",
                           "Attention received, corrupted": f"{noisy[m][f'NCAF attention to {m} (noisy)']:.3f}"}
                          for m in MODS])
    t_faith = pd.DataFrame([{"Explanation used to rank modalities": k,
                             "|Δp(PD)| removing top-ranked": f"{v['delta_p_remove_top']:.3f}",
                             "|Δp(PD)| removing bottom-ranked": f"{v['delta_p_remove_bottom']:.3f}"}
                            for k, v in faith.items()])
    imp = xai_summary["modality_importance"]
    t_imp = pd.DataFrame([{"Modality": MOD_TITLES[m], "Attention A": f"{imp['attention'][k]:.3f}",
                           "Gradient saliency S": f"{imp['saliency'][k]:.3f}",
                           f"Hybrid E (λ={a.lam})": f"{imp['hybrid'][k]:.3f}",
                           "SHAP share (RF)": f"{xai_summary['shap_modality_share'][m]:.3f}"}
                          for k, m in enumerate(MODS)])
    occ_rows = []
    for mode, d in xai_summary["sensor_frequency_occlusion"].items():
        for cls, v in d.items():
            occ_rows.append({"Mode": mode, "Class": cls, **{k: f"{x:+.3f}" for k, x in v.items()}})
    t_occ = pd.DataFrame(occ_rows)
    t_cam = pd.DataFrame([
        {"Input": "Spiral (2-D Grad-CAM)", "Region": "pen ink (dilated)",
         "CAM share in region": f"{xai_summary['gradcam_spiral']['share_on_ink']:.3f}",
         "Region area share": f"{xai_summary['gradcam_spiral']['ink_area_share']:.3f}",
         "Ratio": f"{xai_summary['gradcam_spiral']['ratio']:.2f}"},
        {"Input": "Video (3-D Grad-CAM)", "Region": "moving hand (top-15% temporal std)",
         "CAM share in region": f"{xai_summary['gradcam_video']['share_on_moving_hand']:.3f}",
         "Region area share": f"{xai_summary['gradcam_video']['region_area_share']:.3f}",
         "Ratio": f"{xai_summary['gradcam_video']['ratio']:.2f}"}])
    t_shap = pd.DataFrame([{"Rank": i + 1, "Feature": n, "mean |SHAP|": f"{v:.4f}"}
                           for i, (n, v) in enumerate(xai_summary["shap_top_features"])])
    t_cross = pd.DataFrame([{"Model": n, **{k: f"{100 * v['accuracy']:.1f} / {v['auc']:.3f}" for k, v in r.items()}}
                            for n, r in cross.items()])
    tables = [
        ("E1 - Ablation over fusion strategies (identical frozen encoders, mean ± std over seeds)", t_abl),
        ("E2 - Accuracy (%) with a modality missing at test time", t_miss),
        ("E3 - Accuracy (%) with one modality corrupted by heavy noise", t_noise),
        ("E4 - Cross-attention received by a modality, clean vs corrupted input", t_att),
        ("E5 - Modality importance: attention, saliency (Eq. 6), hybrid (Eq. 7), SHAP", t_imp),
        ("E6 - Faithfulness: mean |Δp(PD)| when the top- vs bottom-ranked modality is removed", t_faith),
        ("E7 - Grad-CAM localisation (ratio > 1 = focuses on the region of interest)", t_cam),
        ("E8 - Sensor frequency occlusion: mean Δp(PD) when a band is removed", t_occ),
        ("E9 - SHAP top-15 handcrafted features (random forest)", t_shap),
        ("E10 - Cross-cohort generalisation, accuracy (%) / AUC on the external cohort", t_cross),
    ]
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        for title, df in tables:
            f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n\n")
            df.to_csv(os.path.join(a.out, title.split(" - ")[0].lower() + ".csv"), index=False)
    json.dump(dict(ablation=abl, missing=miss, noisy=noisy, xai=xai_summary, cross_cohort=cross,
                   end_to_end_reference=e2e), open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = lambda n: os.path.join(a.out, n)  # noqa: E731

    # ablation
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax)
    ax.grid(axis="y", visible=False)
    names_ = list(abl)
    mu = [100 * np.mean([r["accuracy"] for r in abl[n]]) for n in names_]
    sd = [100 * np.std([r["accuracy"] for r in abl[n]]) for n in names_]
    cols = [GRAY if (" only" in n or "Late" in n) else BLUE for n in names_]
    ax.barh(names_, mu, xerr=sd, color=cols, height=0.55, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    for yv, v in enumerate(mu):
        ax.text(v + 1.2, yv, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(50, 102)
    ax.set_xlabel("Test accuracy (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Fusion ablation - gray: no learned fusion, blue: learned fusion heads", fontsize=10,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_e1_ablation.png"), dpi=150)
    plt.close(fig)

    # missing modality: small multiples, one panel per model
    cond_names = list(conds)
    fig, axes = plt.subplots(1, len(miss), figsize=(2.3 * len(miss), 2.9), sharey=True)
    for ax, (n, r) in zip(axes, miss.items()):
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.bar(range(4), [100 * r[c] for c in cond_names], color=[INK] + [MOD_COLORS[m] for m in MODS], width=0.6)
        ax.set_xticks(range(4), ["all", "−sensor", "−spiral", "−video"], fontsize=7, rotation=30)
        ax.set_ylim(40, 100)
        ax.set_title(n.replace(" (", "\n("), fontsize=8, color=INK)
    axes[0].set_ylabel("Accuracy (%)", fontsize=8, color=MUTED)
    fig.suptitle("Accuracy with one modality missing at test time", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out("fig_e2_missing_modality.png"), dpi=150)
    plt.close(fig)

    # attention matrices
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (t, M) in zip(axes, (("All test samples", alpha.mean(0)), ("PD", alpha[lab == 1].mean(0)),
                                 ("Healthy", alpha[lab == 0].mean(0)))):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), [MOD_TITLES[m] for m in MODS], fontsize=8)
        ax.set_yticks(range(3), [MOD_TITLES[m] for m in MODS], fontsize=8)
        ax.set_xlabel("key (attended modality j)", fontsize=8, color=MUTED)
        ax.set_title(t, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("query (modality i)", fontsize=8, color=MUTED)
    fig.suptitle("Cross-modal attention α_ij (Eq. 3), averaged over heads", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out("fig_e3_attention.png"), dpi=150)
    plt.close(fig)

    # modality importance per explanation type + per-patient hybrid
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    _style(ax)
    ax.grid(axis="x", visible=False)
    kinds = [("Attention A", imp["attention"]), ("Saliency S", imp["saliency"]),
             (f"Hybrid E", imp["hybrid"]), ("SHAP (RF)", [xai_summary["shap_modality_share"][m] for m in MODS])]
    w = 0.26
    for k, m in enumerate(MODS):
        ax.bar(np.arange(4) + (k - 1) * w, [v[1][k] for v in kinds], w - 0.03, color=MOD_COLORS[m], label=MOD_TITLES[m])
    ax.set_xticks(range(4), [k[0] for k in kinds], fontsize=8)
    ax.set_ylabel("Share of importance", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Global modality importance", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    ax.grid(axis="x", visible=False)
    order = np.argsort(E[:, 2] - E[:, 0])[:: max(1, len(E) // 60)][:60]
    bottom = np.zeros(len(order))
    for k, m in enumerate(MODS):
        ax.bar(range(len(order)), E[order, k], bottom=bottom, color=MOD_COLORS[m], width=0.8, label=MOD_TITLES[m])
        bottom += E[order, k]
    ax.set_xticks([])
    ax.set_xlabel("60 test patients (sorted)", fontsize=8, color=MUTED)
    ax.set_ylabel("Hybrid explanation E", fontsize=8, color=MUTED)
    ax.set_title(f"Per-patient hybrid explanation E = {a.lam}·A + {1 - a.lam}·S (Eq. 7)", fontsize=9, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_e4_modality_explanations.png"), dpi=150)
    plt.close(fig)

    # Grad-CAM spirals
    pick = list(np.where(lab[:n_cam] == 1)[0][:4]) + list(np.where(lab[:n_cam] == 0)[0][:4])
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for c, i in enumerate(pick):
        axes[0, c].imshow(1 - xi[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"{'PD' if lab[i] else 'Healthy'}  p(PD)={p_s[i]:.2f}", fontsize=8, color=INK)
        axes[1, c].imshow(1 - xi[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(cam_s[i] / (cam_s[i].max() + 1e-9), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, c].axis("off")
    axes[0, 0].text(-8, 32, "input", rotation=90, va="center", fontsize=9, color=MUTED)
    axes[1, 0].text(-8, 32, "Grad-CAM", rotation=90, va="center", fontsize=9, color=MUTED)
    fig.suptitle("Spiral Grad-CAM for the predicted class, through the full cross-attention model", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out("fig_e5_gradcam_spiral.png"), dpi=150)
    plt.close(fig)

    # Grad-CAM videos
    pick = list(np.where(lab[:96] == 1)[0][:3]) + list(np.where(lab[:96] == 0)[0][:3])
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.4))
    raw = T["video"][idx[:96, 2]]
    for c, i in enumerate(pick):
        axes[0, c].imshow(raw[i, 0, 8].numpy(), cmap="gray")
        axes[0, c].set_title(f"{'PD' if lab[i] else 'Healthy'}  p(PD)={p_v[i]:.2f}", fontsize=8, color=INK)
        axes[1, c].imshow(tstd[i], cmap="Blues")
        axes[2, c].imshow(raw[i, 0, 8].numpy(), cmap="gray")
        axes[2, c].imshow(cam_v[i].mean(0) / (cam_v[i].mean(0).max() + 1e-9), cmap="inferno", alpha=0.55)
        for r in range(3):
            axes[r, c].axis("off")
    for r, t in enumerate(["frame 8 (motion)", "temporal std", "3-D Grad-CAM (time-avg)"]):
        axes[r, 0].text(-6, 32, t, rotation=90, va="center", ha="right", fontsize=8, color=MUTED)
    fig.suptitle("Video 3-D Grad-CAM for the predicted class; inputs have the static background removed", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out("fig_e6_gradcam_video.png"), dpi=150)
    plt.close(fig)

    # sensor frequency occlusion + time saliency
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axes[0]
    _style(ax)
    ax.grid(axis="x", visible=False)
    lbl = [f"{b0}-{b1} Hz" for b0, b1 in BANDS]
    for k, (cls, c, col) in enumerate((("PD", 1, BLUE), ("Healthy", 0, ORANGE))):
        ax.bar(np.arange(len(BANDS)) + (k - 0.5) * 0.36, occ_s[lab == c].mean(0), 0.34, color=col, label=cls)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.axvspan(-0.5, 1.5, color="#ecebe6", zorder=0)
    ax.text(0.5, ax.get_ylim()[0] * 0.97, "PD rest-tremor band", ha="center", va="bottom", fontsize=7,
            color=MUTED)
    ax.set_xticks(range(len(BANDS)), lbl, fontsize=8)
    ax.set_ylabel("Δ p(PD) when band removed", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Sensor frequency occlusion (sensor-only mode)", fontsize=9, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    i = int(np.where(lab[:64] == 1)[0][0])
    tt = np.arange(xs.shape[1]) / 50.0
    ch = int(np.abs(tsal[i]).sum(0).argmax())
    ax.plot(tt, xs[i, :, ch].numpy(), lw=1.2, color=GRAY, label=f"input ({['ax', 'ay', 'az', 'gx', 'gy', 'gz'][ch]})")
    ax2 = ax.twinx()
    ax2.fill_between(tt, 0, np.abs(tsal[i, :, ch]), color=BLUE, alpha=0.35, lw=0)
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
    ax.set_title("PD example: input (gray) and |gradient × input| (blue)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_e7_sensor_explanations.png"), dpi=150)
    plt.close(fig)

    # SHAP
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _style(ax)
    ax.grid(axis="y", visible=False)
    tn = [names[i] for i in top][::-1]
    ax.barh(tn, mean_abs[top][::-1], color=[MOD_COLORS[n.split()[0]] for n in tn], height=0.6)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("mean |SHAP value| for p(PD)", fontsize=8, color=MUTED)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MOD_COLORS[m], label=MOD_TITLES[m]) for m in MODS], fontsize=8, frameon=False,
              loc="lower right")
    ax.set_title("SHAP: top-15 handcrafted features (random forest)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_e8_shap.png"), dpi=150)
    plt.close(fig)

    # cross-cohort
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _style(ax)
    ax.grid(axis="x", visible=False)
    strat = ["internal test", "direct transfer", "domain adaptation", "transfer learning"]
    for k, (n, col) in enumerate(zip(cross, (GRAY, ORANGE, BLUE))):
        v = [100 * cross[n][s]["accuracy"] for s in strat]
        ax.bar(np.arange(4) + (k - 1) * 0.27, v, 0.25, color=col, label=n)
    ax.set_xticks(range(4), strat, fontsize=8)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Cross-cohort generalisation (external, domain-shifted cohort)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out("fig_e9_cross_cohort.png"), dpi=150)
    plt.close(fig)

    log("\n" + open(out("tables.md")).read())
    log(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
