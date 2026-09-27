# Replication — Hybrid Multimodal Deep Learning for Parkinson's Disease from Hand Tremor

Full, runnable re-implementation of

> G. Chandra, T. K. Gandhi, B. Singh, **"A Hybrid Multimodal Deep Learning Framework for Robust Diagnosis
> of Parkinson's Disease Using Hand Tremor Analysis"**, *IEEE Sensors Journal*, 26(3):4723–4730, 2026.
> doi:10.1109/JSEN.2025.3641648

Everything in the paper's pipeline (Fig. 3, Algorithm 1) is implemented: three unimodal subnetworks,
late feature-level fusion to a 160-d vector, the hybrid **GRU–LSTNet + BAM** classifier trained end-to-end,
four classical-ML baselines, and every table/figure of the results section (Tables II–V, Figs. 5–6).

```
pd_multimodal_replication/
├── run_replication.py          # the whole experiment, A → Z (Phases 1–6 + baselines + tables + figures)
├── run_extension.py            # extension: explainable AI + cross-modal attention (after NCAF, see end)
├── src/
│   ├── datasets.py             # loaders for the 3 datasets (real formats), preprocessing, subject-wise split, cross-cohort pairing
│   ├── models.py               # SensorLSTM, Spiral2DCNN, Video3DCNN, BAM1D, LSTNet, HybridGRULSTNetBAM, MultimodalNet
│   ├── features.py             # handcrafted features for SVM / RF / k-NN / LR (+ names for SHAP)
│   ├── ncaf.py                 # extension: cross-modal attention fusion with modality masking
│   └── xai.py                  # extension: attention, saliency, hybrid map, Grad-CAM 2-D/3-D, frequency occlusion
├── data/
│   ├── generate_synthetic.py   # seeded simulator producing all three datasets in their real file formats
│   ├── synthetic/              # the generated datasets (committed, ~37 MB)
│   └── synthetic_external/     # domain-shifted external cohort for cross-cohort tests (~14 MB)
├── results/                    # replication: tables.md, table_*.csv, fig5_metrics.png, fig6_confusion_roc.png, log.txt
├── results_extension/          # extension: tables.md, e*.csv, fig_e1…e9 *.png, results.json, log.txt
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python data/generate_synthetic.py          # (optional) re-creates data/synthetic exactly (seed 0)
python run_replication.py --data-root data/synthetic --out results
```
Runs on CPU in about 13 min (4 cores); the end-to-end fusion training dominates.

## About the datasets — please read

The paper uses three public datasets (Table I):

| Paper dataset | Modality | Where to get it |
|---|---|---|
| Dryad Parkinson's Drawing dataset | spiral drawings (x/y, pressure, images) | Dryad / Kaggle "parkinsons-drawings" / UCI "Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet" |
| MPU-9250 hand-tremor sensor dataset (Elsevier) | tri-axial acc + gyro (+mag) CSV | Mendeley Data / Data in Brief |
| PD Motor Severity Estimation dataset (GitHub / MediaPipe) | RGB hand/turning videos, keypoints, UPDRS | GitHub |

The paper gives no URLs/DOIs for them, and the environment this was built in could not reach Dryad, Kaggle,
UCI, Mendeley or Zenodo. **The results in `results/` are therefore produced on the synthetic datasets in
`data/synthetic/`, not on the paper's data, and are not expected to reproduce the paper's exact numbers.**

`data/generate_synthetic.py` writes each dataset **in the same file format as the real one**, so the loaders
that read it are the loaders you use for the real data. It simulates, per subject:
healthy physiological tremor (8–12 Hz, tiny), enhanced physiological tremor in 25 % of controls (5.5–9 Hz),
PD rest tremor (3.5–7 Hz) with amplitude growing with a UPDRS-like severity score 0–4 (score 0 = no visible
tremor), tremor waxing/waning, bradykinesia, micrographia and unstable pen pressure. The class overlap
was calibrated so that single-modality accuracy lands roughly where the paper reports it (≈80–88 %).
The main cohort has 30 PD + 30 control sensor subjects, 40 + 40 spiral subjects (4 drawings each) and
40 + 40 video subjects (8 clips each).

### Using the real data
Put the downloads under one root and run `python run_replication.py --data-root data/real`:
```
data/real/
├── subjects.csv                     # optional: subject,label (0 healthy, 1 PD) for sensor files
├── sensor/*.csv                     # columns ax,ay,az,gx,gy,gz (case-insensitive); label from a `label` column,
│                                    #   subjects.csv, or a PD/HC/parkinson/control token in the path
├── spiral/{healthy,parkinson}/…png  # Kaggle layout, or UCI tablet .txt files under {control,parkinson}/
└── video/{healthy,parkinson}/…mp4   # or a clips.npz (X: N×T×H×W uint8, y, subject); needs opencv-python-headless
```
Sensor sampling rate is assumed to be 100 Hz (`load_sensor(fs=…)` to change).

## What is implemented, mapped to the paper

| Paper | Here |
|---|---|
| Sensor subnetwork: stacked LSTM, x_t ∈ R⁶, h_t ∈ R³², Z_sensor ∈ R³² (Eq. 1) | `SensorLSTM`: LSTM(6→128) → LSTM(128→64) → Dense 32 — **0.12 M params** (Table V: 0.12 M) |
| Spiral subnetwork: Conv2D-ReLU-MaxPool-Flatten-Dense, 64×64×1 → R⁶⁴ (Eqs. 2–3, Fig. 4) | `Spiral2DCNN`: 3×[Conv3×3-ReLU-MaxPool] (32/64/96) → Dense 64 — **0.47 M** (Table V: 0.48 M) |
| Video subnetwork: 3-D CNN, 16×64×64×1 → R⁶⁴ (Eq. 4, Fig. 2) | `Video3DCNN`: 3×[Conv3D-ReLU-MaxPool3D] (16/32/64) → FC 192 → 64 — **3.23 M** (Table V: 3.2 M) |
| Subnetworks trained independently, then fused (Sec. II-B) | each encoder pre-trained with its own softmax head (these are also the Table IV baselines) |
| Z_fused = Concat(…) ∈ R¹⁶⁰ (Eq. 5), late feature-level fusion across cohorts (Sec. II-F) | `MultimodalNet`; fused samples are same-class (sensor, spiral, video) triples, see below |
| BAM → parallel GRU and LSTNet branches → concat(BAM, GRU, LSTNet) → FC → Softmax (Sec. II-G) | `HybridGRULSTNetBAM` with `BAM1D` (channel + dilated spatial attention, F′ = F + F⊙σ(Mc+Ms)) and `LSTNet` (conv → GRU + skip-GRU + AR highway) |
| Z_final = Dropout(ReLU(W·Z_fused+b)), p = 0.5 (Eq. 6); softmax (Eq. 7); CE loss (Eq. 8) | yes |
| Adam, lr 1e-3, 10 epochs, batch 8, validation split 0.2 (Sec. II-H) | yes (all CLI flags) |
| Hold-out test; accuracy, precision, recall, F1 (Eqs. 9–12), confusion matrix, ROC-AUC | yes, plus TPR/TNR |
| SVM (RBF), Random Forest, k-NN (k=5), Logistic Regression (Table III) | yes, on handcrafted tremor features of the same fused triples |
| Inference time per sample and params (Table V) | measured on CPU, batch 1 |

## Where the paper is under-specified, and what was chosen

1. **Cross-cohort fusion.** The three datasets contain *different people* (Sec. II-F says so). A "fused sample"
   can therefore only be a triple of one sensor window, one spiral and one video clip **drawn from the same
   class**. That is what `pair_across_modalities` does. Consequence worth knowing: the fused model is
   combining three *independent* pieces of evidence about the label, so a jump from ~85 % unimodal to ~94 %
   fused is roughly what you'd expect even from simply averaging the three unimodal probabilities
   (reported as an extra row). This is not evidence that the modalities are complementary *within a patient*.
2. **Leakage control.** Train/val/test splits are made **by subject** in every modality, and fused test triples
   are built only from held-out subjects. The paper doesn't say how it split.
3. **How a 160-d vector feeds a GRU / LSTNet / BAM.** The paper does not say. Here Z_fused is treated as a
   length-160 sequence, lifted to 64 channels by a 1-D conv, refined by a 1-D BAM, then fed to a bidirectional
   GRU (128) and an LSTNet branch; BAM output (pooled), GRU state and LSTNet output are concatenated →
   FC 256 → ReLU → Dropout 0.5 → 2-way softmax.
4. **Parameter count of the proposed model.** Table V lists 2.1 M for the proposed model but 3.2 M for the 3-D CNN
   alone, although the proposed model contains the 3-D CNN encoder. This replication's full model has ≈4.2 M
   (encoders 3.8 M + head 0.38 M).
5. **Sensor preprocessing** ("normalisation, cleaning", Fig. 3): 4th-order Butterworth high-pass at 3 Hz,
   decimation 100 → 50 Hz, 2.56 s windows (T = 128) with 50 % overlap, one global scale per channel.
   Per-recording z-scoring was tried first and removed the tremor-amplitude information (LSTM ≈ 57 %).
6. **LSTM readout.** Algorithm 1 uses Z_sensor = Dense(h_T). With the paper's 10 epochs / batch 8 that readout
   collapsed to chance on 2 of 3 seeds in our tests; mean-pooling the LSTM states over time was stable
   (85–89 %). Mean-pooling is the default; `--lstm-readout last` restores the paper's version.
7. **Gradient-norm clipping (1.0)** is used in all deep models for stability (not mentioned in the paper).
8. **Model selection.** The weights from the epoch with the best validation accuracy are kept.
9. **Classical baselines' inputs** are not stated; they get handcrafted tremor features (Welch band powers,
   dominant frequency, spiral radial/roughness statistics, video motion-energy spectra) of the *same fused
   triples*, so every row of Tables III/IV is scored on the identical 600-triple test set.

## Results (synthetic data, seed 42)

All numbers are on the **synthetic** datasets (see above), on one shared test set of 600 fused triples
built only from held-out subjects. Full tables: [`results/tables.md`](results/tables.md); figures:
[`results/fig5_metrics.png`](results/fig5_metrics.png), [`results/fig6_confusion_roc.png`](results/fig6_confusion_roc.png).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Paper accuracy |
|---|---|---|---|---|---|---|
| SVM (RBF) | 95.3 % | 92.0 % | 99.3 % | 95.5 % | 0.998 | 75.3 % |
| Random Forest | 95.5 % | 91.7 % | 100 % | 95.7 % | 0.997 | 78.1 % |
| k-NN (k=5) | 98.5 % | 99.3 % | 97.7 % | 98.5 % | 0.998 | 70.4 % |
| Logistic Regression | 94.2 % | 90.5 % | 98.7 % | 94.4 % | 0.995 | 72.8 % |
| Sensor – LSTM | 80.0 % | 87.5 % | 70.0 % | 77.8 % | 0.880 | 84.5 % |
| Spiral – 2-D CNN | 82.8 % | 91.2 % | 72.7 % | 80.9 % | 0.886 | 81.2 % |
| Video – 3-D CNN | 87.5 % | 93.4 % | 80.7 % | 86.6 % | 0.931 | 86.0 % |
| **Proposed GRU-LSTNet + BAM (end-to-end, paper protocol)** | **84.3 %** | 87.6 % | 80.0 % | 83.6 % | 0.947 | **94.0 %** |
| Averaging the three unimodal probabilities (no learned fusion) | 91.8 % | 94.0 % | 89.3 % | 91.6 % | 0.983 | – |

What this says:

1. **The unimodal subnetworks replicate** — sizes match Table V and accuracies land in the paper's range.
2. **The paper's headline result does not replicate here.** Trained end-to-end exactly as described
   (10 epochs, batch 8, lr 1e-3), the fused model overfits: training loss reaches 0.02 while validation
   accuracy swings between 75 % and 93 %. It ends *below* the video model alone and below plain probability
   averaging. The fix is in the extension below: with the encoders frozen, the same head reaches 94.9 %.
3. **Classical ML wins on this data**, but that is a synthetic-data caveat: the simulator generates tremor as
   exactly the band-power / dominant-frequency signal those features measure. Do not read it as a claim about
   real patients.
4. Table V: the proposed model has 4.2 M parameters and 33.7 ms CPU inference per sample (paper: 2.1 M, 7.6 ms
   on unspecified hardware).

CPU training is multithreaded and not bit-for-bit deterministic, so re-running gives slightly different
numbers (±1–3 points for the deep models).

---

# Extension — explainable AI and cross-modal attention (after NCAF, Gujjeti et al. 2026)

> S. Gujjeti et al., **"NeuroCrossAttention fusion for multimodal explainable early diagnosis of
> neurodegenerative diseases"**, *Discover Computing* 29:329, 2026. doi:10.1007/s10791-026-10200-2

This part is **not in the replicated paper**. It adds the NCAF paper's explainability module and
cross-modal attention fusion to the tremor model. Code: [`src/xai.py`](src/xai.py),
[`src/ncaf.py`](src/ncaf.py), [`run_extension.py`](run_extension.py). Output:
[`results_extension/`](results_extension/) (all tables in [`tables.md`](results_extension/tables.md)).

```bash
python data/generate_synthetic.py --out data/synthetic_external --seed 1 --shift 1.0 \
    --sensor-subjects 20 --spiral-subjects 20 --video-subjects 15 --clips-per-subject 6   # (already committed)
python run_extension.py            # ~25 min on 4 CPU cores
```

**Model explained:** three frozen pre-trained encoders → cross-modal attention fusion
(α_ij = softmax(Q_i K_jᵀ/√d), H_i = Σ_j α_ij V_j, F = [H_sensor‖H_spiral‖H_video], NCAF Eqs. 3–5) trained with
modality masking (p = 0.2). 95.4 % test accuracy.

The NCAF paper checks its explanations only by eye. Here the synthetic data has a **known ground truth**
(PD rest tremor 3.5–7 Hz; enhanced physiological tremor in controls 5.5–9 Hz; the tremor is in the pen line
and the moving hand), so each explanation is **scored**.

## Explainability results

| Method (NCAF reference) | What it shows | Result | Verdict |
|---|---|---|---|
| Sensor frequency occlusion | Δp(PD) when a band is removed from the sensor window | PD patients: removing **3–5 Hz → −0.46**, 5–7 Hz → −0.11, bands above 7 Hz ≈ 0. Controls: largest drop at **7–9 Hz (−0.08)** | ✅ matches the simulated ground truth exactly |
| SHAP on handcrafted features (tabular) | top features of the random forest | top 5 = gyro **dominant frequency** (x, y, z) and gyro **4–6.5 Hz band power** | ✅ matches ground truth |
| Cross-modal attention α (Eq. 3) | which modality each modality attends to | sensor receives 0.47 of attention, video 0.29, spiral 0.25 | consistent with SHAP and saliency (sensor ≈ 0.5 in all three) |
| Hybrid explanation E = λA + (1−λ)S (Eq. 7) | per-patient modality contributions | see faithfulness below | – |
| Faithfulness (deletion test) | \|Δp\| when removing the top- vs bottom-ranked modality | attention **0.173** vs 0.017 · hybrid 0.164 vs 0.023 · saliency 0.130 vs 0.025 · random 0.068 vs 0.069 | ✅ all explanations are faithful (≈2.5× random); **the hybrid does not beat attention alone** |
| 2-D Grad-CAM (spiral) | heat on pen ink vs background | 50 % of CAM mass on ink that covers 41 % of the image (ratio **1.22**) | ⚠️ weak: several PD spirals are explained by image borders |
| 3-D Grad-CAM (video) | heat on the moving hand | 23 % of CAM mass on a region covering 15 % (ratio **1.55**) | ⚠️ weak: often diffuse or empty |
| Time saliency (Eq. 6, sensor) | \|gradient × input\| over time | concentrated in the first samples of the window | ⚠️ LSTM boundary artefact; frequency occlusion is the better sensor explanation |

Figures:
[attention](results_extension/fig_e3_attention.png) ·
[modality explanations](results_extension/fig_e4_modality_explanations.png) ·
[spiral Grad-CAM](results_extension/fig_e5_gradcam_spiral.png) ·
[video Grad-CAM](results_extension/fig_e6_gradcam_video.png) ·
[sensor explanations](results_extension/fig_e7_sensor_explanations.png) ·
[SHAP](results_extension/fig_e8_shap.png)

**Take-aways for explainability**
- Explanations defined on **signal properties** (frequency occlusion, SHAP on spectral features) recovered the
  true tremor mechanism. Pixel-level Grad-CAM, the NCAF paper's main clinical visual, was the least reliable.
  Showing a heatmap is not evidence that it is right; it has to be scored against a known ground truth.
- The NCAF hybrid map (Eq. 7) was not more faithful than attention weights alone in this setting.
- The NCAF claim that *"the noisier an input is, the lower the attention weights it receives"* **did not hold**:
  corrupting the sensor with heavy noise *raised* the attention it received (0.47 → 0.54), and every model
  dropped to ≈ 52 %. Attention weights should not be read as a reliability score.

## Secondary results (fusion ablation, robustness, cross-cohort)

**Fusion ablation, identical frozen encoders, 5 seeds** ([fig](results_extension/fig_e1_ablation.png)):

| Configuration | Accuracy | ROC-AUC |
|---|---|---|
| Best single modality (3-D CNN) | 87.8 % | 0.892 |
| Late fusion (probability averaging) | 93.5 % | 0.979 |
| Concat + MLP | 95.4 ± 0.2 % | 0.988 |
| Concat + MLP + masking | 95.6 ± 0.2 % | 0.989 |
| GRU-LSTNet + BAM (paper head) | 94.9 ± 1.5 % | 0.985 |
| Cross-attention (NCAF) | 95.4 ± 0.4 % | 0.984 |
| Cross-attention + masking | 95.4 ± 0.7 % | 0.987 |
| *GRU-LSTNet + BAM trained end-to-end (main replication)* | *84.3 %* | *0.947* |

The fusion architecture barely matters (all within about 1 point); **freezing the encoders** matters most.

**Missing modality at test time** ([fig](results_extension/fig_e2_missing_modality.png)): modality masking makes
plain concatenation robust (without the sensor: 85.2 % → 93.5 %). Cross-attention handles a missing input
reasonably even without masking (≥ 92 %).

**Cross-cohort generalisation** ([fig](results_extension/fig_e9_cross_cohort.png)): the external cohort
has different subjects and a shifted device/site (sensor noise and gain, pen, lighting, hand size).

| Model | Internal test | Direct transfer | Domain adaptation (unsupervised) | Transfer learning (30 % labelled) |
|---|---|---|---|---|
| Concat + MLP | 95.0 % | 83.5 % | 84.0 % | 88.2 % |
| GRU-LSTNet + BAM | 95.8 % | 86.2 % | 85.2 % | 90.8 % |
| Cross-attention + masking | 95.3 % | 84.8 % | 86.0 % | 90.0 % |

All models lose about 10 points under domain shift. Fine-tuning the head on 30 % of external subjects recovers
about half of that. Simple embedding alignment helps little.
