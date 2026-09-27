### Table II - Performance metrics of the proposed model

| Metric    | Value   |
|:----------|:--------|
| Accuracy  | 84.33%  |
| Precision | 87.59%  |
| Recall    | 80.00%  |
| F1-Score  | 83.62%  |

### Table III - Comparison with machine learning models

| Model               | Accuracy   | Precision   | Recall   | F1-Score   |   ROC-AUC | TPR     | TNR    |
|:--------------------|:-----------|:------------|:---------|:-----------|----------:|:--------|:-------|
| SVM (RBF)           | 95.33%     | 91.98%      | 99.33%   | 95.51%     |     0.998 | 99.33%  | 91.33% |
| Random Forest       | 95.50%     | 91.74%      | 100.00%  | 95.69%     |     0.997 | 100.00% | 91.00% |
| k-NN (k=5)          | 98.50%     | 99.32%      | 97.67%   | 98.49%     |     0.998 | 97.67%  | 99.33% |
| Logistic Regression | 94.17%     | 90.52%      | 98.67%   | 94.42%     |     0.995 | 98.67%  | 89.67% |
| Proposed Model      | 84.33%     | 87.59%      | 80.00%   | 83.62%     |     0.947 | 80.00%  | 88.67% |

### Table IV - Comparison across modalities

| Modality   | Model          | Accuracy   | Precision   | Recall   | F1-Score   |   ROC-AUC | TPR    | TNR    |
|:-----------|:---------------|:-----------|:------------|:---------|:-----------|----------:|:-------|:-------|
| Sensor     | LSTM           | 80.00%     | 87.50%      | 70.00%   | 77.78%     |     0.879 | 70.00% | 90.00% |
| Spiral     | 2D CNN         | 82.83%     | 91.21%      | 72.67%   | 80.89%     |     0.886 | 72.67% | 93.00% |
| Video      | 3D CNN         | 87.50%     | 93.44%      | 80.67%   | 86.58%     |     0.931 | 80.67% | 94.33% |
| Fused      | Proposed Model | 84.33%     | 87.59%      | 80.00%   | 83.62%     |     0.947 | 80.00% | 88.67% |

### Table V - Inference time per sample (CPU, batch 1) and model complexity

| Model               |   Inference time (ms) | Model type          | Params (M)   |
|:--------------------|----------------------:|:--------------------|:-------------|
| SVM (RBF)           |                  0.62 | Classical ML        | -            |
| Random Forest       |                 65.27 | Ensemble ML         | -            |
| k-NN (k=5)          |                  0.84 | Instance-based ML   | -            |
| Logistic Regression |                  0.47 | Linear ML           | -            |
| LSTM                |                  1.71 | RNN-based DL        | 0.12         |
| 2D CNN              |                  1.46 | Spatial CNN         | 0.47         |
| 3D CNN              |                 10.53 | Spatio-temporal CNN | 3.23         |
| Proposed Model      |                 33.65 | GRU-LSTNet + BAM    | 4.20         |

### Extra - probability-averaging reference and unimodal scores on their own test splits

| Model                          |   accuracy |   precision |   recall |     f1 |    auc |    tnr |    tpr |
|:-------------------------------|-----------:|------------:|---------:|-------:|-------:|-------:|-------:|
| Unimodal probability averaging |     0.9183 |      0.9404 |   0.8933 | 0.9162 | 0.9825 | 0.9433 | 0.8933 |
| LSTM (own test split)          |     0.7828 |      0.8684 |   0.6667 | 0.7543 | 0.8601 | 0.899  | 0.6667 |
| 2D CNN (own test split)        |     0.8125 |      0.875  |   0.7292 | 0.7955 | 0.8841 | 0.8958 | 0.7292 |
| 3D CNN (own test split)        |     0.8594 |      0.9259 |   0.7812 | 0.8475 | 0.9183 | 0.9375 | 0.7812 |

