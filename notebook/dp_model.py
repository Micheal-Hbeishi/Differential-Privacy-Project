from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baseline_model import (
    DATA_PATH,
    DROP_COLUMNS,
    RESULTS_PATH,
    TARGET_COLUMN,
    build_model,
    load_data,
)


DEFAULT_EPSILON = 10.0
DEFAULT_TARGET_BOUNDS = (0.0, 2500.0)


class NoisyRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha=1.0,
        epsilon=DEFAULT_EPSILON,
        target_bounds=DEFAULT_TARGET_BOUNDS,
        random_state=42,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.target_bounds = target_bounds
        self.random_state = random_state

    def fit(self, X, y):
        lower, upper = self.target_bounds
        clipped_y = np.clip(y, lower, upper)

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, clipped_y)

        rng = np.random.default_rng(self.random_state)
        y_range = upper - lower
        noise_scale = y_range / (max(self.epsilon, 1e-12) * len(clipped_y))

        self.coef_ = self.model_.coef_ + rng.laplace(
            loc=0.0,
            scale=noise_scale,
            size=self.model_.coef_.shape,
        )
        self.intercept_ = self.model_.intercept_ + rng.laplace(
            loc=0.0,
            scale=noise_scale,
        )
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def build_dp_model(X, epsilon=DEFAULT_EPSILON, target_bounds=DEFAULT_TARGET_BOUNDS):
    baseline_pipeline = build_model(X)
    preprocessor = baseline_pipeline.named_steps["preprocessor"]

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                NoisyRidge(
                    epsilon=epsilon,
                    target_bounds=target_bounds,
                    random_state=42,
                ),
            ),
        ]
    )


def train_dp_model(
    epsilon=DEFAULT_EPSILON,
    target_bounds=DEFAULT_TARGET_BOUNDS,
    data_path=DATA_PATH,
):
    df = load_data(data_path)
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_dp_model(X_train, epsilon=epsilon, target_bounds=target_bounds)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": "dp_noisy_ridge_regression",
        "epsilon": epsilon,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return model, metrics


def save_dp_metrics(metrics, path=RESULTS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([metrics])
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        existing = existing[existing["model"] != metrics["model"]]
        new_row = pd.concat([existing, new_row], ignore_index=True)

    new_row.to_csv(path, index=False)


if __name__ == "__main__":
    _, dp_metrics = train_dp_model()
    save_dp_metrics(dp_metrics)

    print("DP-style model trained")
    for name, value in dp_metrics.items():
        print(f"{name}: {value}")
