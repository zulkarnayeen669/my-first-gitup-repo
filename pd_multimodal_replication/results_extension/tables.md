### E1 - Ablation over fusion strategies (identical frozen encoders, mean ± std over seeds)

| Configuration                                   |   Seeds | Accuracy (%)   | F1 (%)       | Recall (%)   |   ROC-AUC |
|:------------------------------------------------|--------:|:---------------|:-------------|:-------------|----------:|
| LSTM only                                       |       1 | 87.17          | 86.03        | 79.00        |     0.866 |
| 2D CNN only                                     |       1 | 77.00          | 77.88        | 81.00        |     0.9   |
| 3D CNN only                                     |       1 | 87.83          | 87.48        | 85.00        |     0.892 |
| Late fusion (probability averaging)             |       1 | 93.50          | 93.33        | 91.00        |     0.979 |
| Concat + MLP                                    |       5 | 95.37 ± 0.22   | 95.37 ± 0.23 | 95.40 ± 0.44 |     0.988 |
| Concat + MLP + masking                          |       5 | 95.63 ± 0.24   | 95.65 ± 0.25 | 95.93 ± 0.49 |     0.989 |
| GRU-LSTNet + BAM (paper head)                   |       5 | 94.93 ± 1.51   | 94.87 ± 1.62 | 94.20 ± 2.84 |     0.985 |
| Cross-attention (NCAF)                          |       5 | 95.37 ± 0.39   | 95.35 ± 0.41 | 95.13 ± 0.91 |     0.984 |
| Cross-attention + masking                       |       5 | 95.43 ± 0.71   | 95.42 ± 0.73 | 95.13 ± 1.13 |     0.987 |
| GRU-LSTNet + BAM, end-to-end (main replication) |       1 | 84.33          | 83.62        | 80.00        |     0.947 |

### E2 - Accuracy (%) with a modality missing at test time

| Model                               |   all |   no sensor |   no spiral |   no video |
|:------------------------------------|------:|------------:|------------:|-----------:|
| Concat + MLP                        |  95   |        85.2 |        91.2 |       90.8 |
| Concat + MLP + masking              |  95.8 |        93.5 |        93.5 |       92.3 |
| GRU-LSTNet + BAM (paper head)       |  95.8 |        93.7 |        94.3 |       90.3 |
| Cross-attention (NCAF)              |  94.7 |        92   |        94.3 |       92.7 |
| Cross-attention + masking           |  95.3 |        90.2 |        92.8 |       92.7 |
| Late fusion (probability averaging) |  93.5 |        85   |        87.3 |       90.7 |

### E3 - Accuracy (%) with one modality corrupted by heavy noise

| Model                         |   Sensor corrupted |   Spiral corrupted |   Video corrupted |
|:------------------------------|-------------------:|-------------------:|------------------:|
| Concat + MLP                  |               51.5 |               94.5 |              93.5 |
| Concat + MLP + masking        |               51.5 |               95.7 |              95.2 |
| GRU-LSTNet + BAM (paper head) |               52   |               95.7 |              95.5 |
| Cross-attention (NCAF)        |               51.8 |               94.7 |              93.5 |
| Cross-attention + masking     |               51.8 |               95   |              94.8 |

### E4 - Cross-attention received by a modality, clean vs corrupted input

| Modality   |   Attention received, clean |   Attention received, corrupted |
|:-----------|----------------------------:|--------------------------------:|
| Sensor     |                       0.47  |                           0.543 |
| Spiral     |                       0.246 |                           0.235 |
| Video      |                       0.285 |                           0.277 |

### E5 - Modality importance: attention, saliency (Eq. 6), hybrid (Eq. 7), SHAP

| Modality   |   Attention A |   Gradient saliency S |   Hybrid E (λ=0.5) |   SHAP share (RF) |
|:-----------|--------------:|----------------------:|-------------------:|------------------:|
| Sensor     |         0.47  |                 0.557 |              0.513 |             0.495 |
| Spiral     |         0.246 |                 0.223 |              0.235 |             0.407 |
| Video      |         0.285 |                 0.22  |              0.252 |             0.098 |

### E6 - Faithfulness: mean |Δp(PD)| when the top- vs bottom-ranked modality is removed

| Explanation used to rank modalities   |   |Δp(PD)| removing top-ranked |   |Δp(PD)| removing bottom-ranked |
|:--------------------------------------|-------------------------------:|----------------------------------:|
| attention only (lambda=1)             |                          0.173 |                             0.017 |
| hybrid (lambda=0.5)                   |                          0.164 |                             0.023 |
| saliency only (lambda=0)              |                          0.13  |                             0.025 |
| random ranking                        |                          0.068 |                             0.069 |

### E7 - Grad-CAM localisation (ratio > 1 = focuses on the region of interest)

| Input                 | Region                             |   CAM share in region |   Region area share |   Ratio |
|:----------------------|:-----------------------------------|----------------------:|--------------------:|--------:|
| Spiral (2-D Grad-CAM) | pen ink (dilated)                  |                 0.5   |               0.409 |    1.22 |
| Video (3-D Grad-CAM)  | moving hand (top-15% temporal std) |                 0.233 |               0.15  |    1.55 |

### E8 - Sensor frequency occlusion: mean Δp(PD) when a band is removed

| Mode           | Class   |   3-5 Hz |   5-7 Hz |   7-9 Hz |   9-12 Hz |   12-25 Hz |
|:---------------|:--------|---------:|---------:|---------:|----------:|-----------:|
| sensor only    | PD      |   -0.462 |   -0.11  |   -0.003 |     0.003 |      0.013 |
| sensor only    | healthy |   -0.028 |   -0.055 |   -0.077 |     0.001 |      0.024 |
| all modalities | PD      |   -0.098 |   -0.036 |   -0.001 |     0     |      0.004 |
| all modalities | healthy |   -0.024 |   -0.033 |   -0.048 |     0.011 |      0.031 |

### E9 - SHAP top-15 handcrafted features (random forest)

|   Rank | Feature                    |   mean |SHAP| |
|-------:|:---------------------------|--------------:|
|      1 | sensor gy dominant freq    |        0.0387 |
|      2 | sensor gx dominant freq    |        0.0319 |
|      3 | sensor gz dominant freq    |        0.0292 |
|      4 | sensor gy power 4-6.5Hz    |        0.0282 |
|      5 | sensor gz power 4-6.5Hz    |        0.0208 |
|      6 | spiral FFT (1,0)           |        0.0205 |
|      7 | spiral mean radius         |        0.0178 |
|      8 | sensor gx power 4-6.5Hz    |        0.0172 |
|      9 | spiral FFT (1,1)           |        0.0154 |
|     10 | video moving-area fraction |        0.0152 |
|     11 | sensor ay dominant freq    |        0.0146 |
|     12 | spiral FFT (6,0)           |        0.0143 |
|     13 | spiral FFT (6,1)           |        0.014  |
|     14 | spiral FFT (4,1)           |        0.0131 |
|     15 | sensor gx zero-cross rate  |        0.0129 |

### E10 - Cross-cohort generalisation, accuracy (%) / AUC on the external cohort

| Model                         | internal test   | direct transfer   | domain adaptation   | transfer learning   |
|:------------------------------|:----------------|:------------------|:--------------------|:--------------------|
| Concat + MLP                  | 95.0 / 0.988    | 83.5 / 0.928      | 84.0 / 0.941        | 88.2 / 0.956        |
| GRU-LSTNet + BAM (paper head) | 95.8 / 0.989    | 86.2 / 0.928      | 85.2 / 0.948        | 90.8 / 0.974        |
| Cross-attention + masking     | 95.3 / 0.986    | 84.8 / 0.919      | 86.0 / 0.932        | 90.0 / 0.974        |

