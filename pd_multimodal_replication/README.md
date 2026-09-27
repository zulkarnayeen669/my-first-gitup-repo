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
├── src/
│   ├── datasets.py             # loaders for the 3 datasets (real formats), preprocessing, subject-wise split, cross-cohort pairing
│   ├── models.py               # SensorLSTM, Spiral2DCNN, Video3DCNN, BAM1D, LSTNet, HybridGRULSTNetBAM, MultimodalNet
│   └── features.py             # handcrafted features for SVM / RF / k-NN / LR
├── data/
│   ├── generate_synthetic.py   # seeded simulator producing all three datasets in their real file formats
│   └── synthetic/              # the generated datasets (committed, ~27 MB)
├── results/                    # tables.md, table_*.csv, fig5_metrics.png, fig6_confusion_roc.png, results.json, log.txt
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python data/generate_synthetic.py          # (optional) re-creates data/synthetic exactly (seed 0)
python run_replication.py --data-root data/synthetic --out results
```
Runs on CPU in about 15–25 min (4 cores); the 3-D CNN dominates.

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

RESULTS_PLACEHOLDER
