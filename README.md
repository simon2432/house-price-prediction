# House Price Prediction — Ames Housing Dataset

A complete supervised machine learning pipeline that predicts residential house sale prices using the [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset). The project walks through every stage of the ML workflow — from raw data exploration to a final comparison of six models — and is designed to be readable, reproducible, and ready to run.

---

## Dataset

| Property        | Value                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Source          | [Kaggle — Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) |
| Rows            | ~2,930                                                                                         |
| Raw features    | ~80 (categorical + numerical)                                                                  |
| Target variable | `SalePrice` (USD)                                                                              |

---

## Workflow

1. **Exploratory Data Analysis** — price distribution, feature correlations, scatter plots, missing value audit
2. **Data Cleaning** — drop administrative identifiers, remove columns with >50% missing data, IQR outlier removal on the target
3. **Feature Engineering** — six domain-driven features: `house_age`, `years_since_remodel`, `total_bath`, `total_porch_sf`, `has_garage`, `overall_score`
4. **Preprocessing Pipeline** — `ColumnTransformer` wrapping median imputation + standard scaling for numerics, and most-frequent imputation + one-hot encoding for categoricals; always refitted inside cross-validation folds to prevent leakage
5. **Classical Regression Models** — Linear Regression, Ridge (tuned), Lasso (tuned), Random Forest, Gradient Boosting
6. **5-Fold Cross-Validation** — unbiased performance estimates across all classical models
7. **Neural Network** — Keras MLP with Dropout regularisation, scaled target (StandardScaler), and EarlyStopping
8. **Final Comparison** — MAE, RMSE, R² on the held-out test set, with actual-vs-predicted plots and metric bar charts

---

## Models & Tuning

| Model             | Configuration                                                    |
| ----------------- | ---------------------------------------------------------------- |
| Linear Regression | Baseline — no regularisation                                     |
| Ridge             | `GridSearchCV` over α ∈ {0.1, 1, 10, 100, 500, 1000}             |
| Lasso             | `GridSearchCV` over α ∈ {0.01, 0.1, 1, 10, 100}                  |
| Random Forest     | 300 estimators, `max_features='sqrt'`, parallel                  |
| Gradient Boosting | 200 estimators, lr=0.05, max_depth=4                             |
| Neural Network    | 256 → 128 → 64 → 1, Dropout(0.3/0.2), EarlyStopping(patience=20) |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebook.ipynb
```

---

## Project Structure

```
house-price-prediction/
├── data/
│   └── AmesHousing.csv      # Raw dataset
├── notebook.ipynb           # Full ML pipeline
├── requirements.txt         # Pinned dependencies
└── README.md
```

---

## Results

Metrics on the held-out test set (20% of the data, `random_state=42`).

| Model              | MAE         | RMSE        | R²         |
| ------------------ | ----------- | ----------- | ---------- |
| Linear Regression  | $13,376     | $25,328     | 0.8275     |
| Ridge (α=10)       | $13,872     | $23,169     | 0.8557     |
| Lasso (α=100)      | $13,614     | $23,223     | 0.8550     |
| Random Forest      | $14,000     | $20,394     | 0.8882     |
| Gradient Boosting  | $12,273     | $17,893     | 0.9139     |
| **Neural Network** | **$12,913** | **$17,615** | **0.9166** |

5-fold cross-validation on the training set (mean ± std):

| Model             | CV R²          | CV RMSE          |
| ----------------- | -------------- | ---------------- |
| Linear Regression | 0.8317 ± 0.106 | $22,985 ± $6,519 |
| Ridge             | 0.8453 ± 0.103 | $21,920 ± $6,216 |
| Lasso             | 0.8495 ± 0.104 | $21,553 ± $6,344 |
| Random Forest     | 0.8766 ± 0.017 | $20,434 ± $1,169 |
| Gradient Boosting | 0.8983 ± 0.022 | $18,486 ± $1,723 |

The ensemble methods (Random Forest, Gradient Boosting) are notably more stable across folds (std ≈ 0.02) compared to the linear models (std ≈ 0.10). The neural network stopped training at epoch 31 out of 300 maximum epochs thanks to early stopping.

---

## Requirements

- Python 3.9+
- See `requirements.txt` for the full pinned dependency list
- A GPU is not required; the neural network trains in under a minute on CPU
