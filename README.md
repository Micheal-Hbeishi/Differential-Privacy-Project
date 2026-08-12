# Differential Privacy, Utility, and Explainability Project

This repository now includes a **paper-aligned image-classification replication** of Abbasi, Mori, and Saracino (IEEE TDSC, 2025), alongside the original introductory sales-regression experiment.

For academic review, start with [PROFESSOR_REPORT.md](PROFESSOR_REPORT.md) and run:

```bash
python notebook/paper_aligned_replication.py
```

The replication uses handwritten-digit images, neural classification, per-example clipped and Gaussian-noised DP-SGD, a bottleneck autoencoder, SmoothGrad explanations, and a privacy-utility-explainability compatibility matrix. Its scope and deviations from the original paper are explicitly documented; it does not claim exact numerical reproduction or a production-certified privacy guarantee.

## Legacy tabular experiment

This project compares three Ridge regression approaches for predicting chocolate sales amounts:

- A non-private baseline Ridge regression model.
- A DP-style noisy Ridge regression model that clips targets and adds Laplace noise to learned coefficients.
- An adaptive DP-style noisy Ridge regression model that allocates more privacy budget to stronger learned coefficients.

The goal is to explore the accuracy tradeoff introduced by privacy-inspired noise while keeping the preprocessing and train/test split consistent across models.

## Project Structure

```
.
├── notebook/
│   ├── baseline_model.py        # Baseline Ridge regression pipeline
│   ├── dp_model.py              # Fixed-budget DP-style noisy Ridge model
│   ├── adaptive_dp_model.py     # Adaptive-budget DP-style noisy Ridge model
│   └── dp_experiment.ipynb      # Experiment notebook
├── results/
│   ├── evaluate.py              # Prints model comparison metrics
│   └── metrics_table.csv        # Saved model results
├── src/
│   └── Chocolate_Sales.csv      # Chocolate sales dataset
├── requirements.txt
└── README.md
```

## Dataset

The included dataset contains chocolate sales records with fields such as product, country, sales channel, salesperson, order date, discount percentage, price per box, marketing spend, boxes shipped, and sale amount.

The prediction target is:

```
Amount

```

The modeling scripts clean numeric columns, extract date features from `Order_Date`, one-hot encode categorical features, scale numeric features, and evaluate performance on a fixed 80/20 train/test split.

## Models

### Baseline Ridge Regression

`notebook/baseline_model.py` trains a standard Ridge regression model with shared preprocessing.

### DP-Style Noisy Ridge Regression

`notebook/dp_model.py` clips target values to a fixed range and adds Laplace noise to the fitted Ridge coefficients and intercept. The default privacy budget is:

```
epsilon = 10.0
```

### Adaptive DP-Style Noisy Ridge Regression

`notebook/adaptive_dp_model.py` uses the same target clipping and Laplace noise approach, but distributes the feature privacy budget based on absolute coefficient magnitude. Larger coefficients receive more epsilon, which reduces their noise scale.

> Note: These scripts are intended as an educational privacy experiment. They use DP-inspired mechanisms, but they should not be treated as a production privacy guarantee without formal sensitivity analysis and privacy accounting.

## Results

Current saved results:

| Model                                    | MAE      | RMSE     | R2   | Epsilon |
|-----------------------------------------:|---------:|---------:|:----:|--------:|
| Baseline Ridge Regression                | 107.6727 | 207.3698 | 0.6902 |      |
| DP-Style Noisy Ridge Regression          | 107.6891 | 208.2328 | 0.6876 | 10.0 |
| Adaptive DP-Style Noisy Ridge Regression | 107.6702 | 208.2565 | 0.6875 | 10.0 |

Lower MAE and RMSE are better. Higher R2 is better.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the models

Run the scripts from the project root:

```bash
python notebook/baseline_model.py
python notebook/dp_model.py
python notebook/adaptive_dp_model.py
```

### 5. Compare results

```bash
python results/evaluate.py
```

## Reproducibility

Each model uses `random_state=42` for stable train/test splits and repeatable noise generation. Re-running the scripts in the order above will refresh `results/metrics_table.csv`.

## Possible Extensions

- Experiment with smaller epsilon values to increase privacy noise.
- Add cross-validation across multiple train/test splits.
- Plot privacy-utility tradeoffs across several epsilon values.
- Replace the educational DP-style mechanism with a formally analyzed DP training method.
- Add tests for preprocessing, metric generation, and model output shape.
