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


# Total privacy budget. Larger epsilon means less noise and weaker privacy.
DEFAULT_EPSILON = 10.0

# Target values are clipped into this range before fitting so one extreme sale
# cannot have unlimited influence on the model.
DEFAULT_TARGET_BOUNDS = (0.0, 2500.0)

# Save part of the privacy budget for the intercept, which is the model's
# baseline prediction when all feature values are zero.
DEFAULT_INTERCEPT_EPSILON_FRACTION = 0.1

# Give every feature at least a small privacy budget, even if its learned
# coefficient is tiny.
DEFAULT_MIN_FEATURE_EPSILON_FRACTION = 0.05


class AdaptiveNoisyRidge(BaseEstimator, RegressorMixin):
    """Ridge regression with adaptive DP-style noise added to the coefficients.

    The model first trains a normal Ridge regression on clipped target values.
    Then it adds Laplace noise to the learned coefficients. Unlike the regular
    DP model, this version gives more privacy budget to features that appear
    more important, so those features receive less noise.
    """

    def __init__(
        self,
        alpha=1.0,
        epsilon=DEFAULT_EPSILON,
        target_bounds=DEFAULT_TARGET_BOUNDS,
        intercept_epsilon_fraction=DEFAULT_INTERCEPT_EPSILON_FRACTION,
        min_feature_epsilon_fraction=DEFAULT_MIN_FEATURE_EPSILON_FRACTION,
        random_state=42,
    ):
        # alpha controls how strongly Ridge regression shrinks coefficients.
        self.alpha = alpha

        # epsilon is the total privacy budget used by this adaptive model.
        self.epsilon = epsilon

        # target_bounds controls how low/high the sale amount can be before
        # training. Clipping is a common DP step because it limits sensitivity.
        self.target_bounds = target_bounds

        # This fraction of epsilon is reserved for the intercept noise.
        self.intercept_epsilon_fraction = intercept_epsilon_fraction

        # This fraction of the feature budget is spread evenly across all
        # features before the adaptive allocation happens.
        self.min_feature_epsilon_fraction = min_feature_epsilon_fraction

        # Fixed seed makes results repeatable each time you run the file.
        self.random_state = random_state

    def fit(self, X, y):
        """Train the Ridge model, then add adaptive privacy noise."""
        lower, upper = self.target_bounds

        # Clip sale amounts so extreme rows cannot dominate the training result.
        clipped_y = np.clip(y, lower, upper)

        # Train the normal Ridge model first. The noisy model starts from these
        # learned coefficients.
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, clipped_y)

        # Get the learned feature weights and decide how much epsilon each
        # feature receives.
        coef = self.model_.coef_
        feature_epsilons = self._allocate_feature_epsilons(coef)

        rng = np.random.default_rng(self.random_state)
        y_range = upper - lower

        # Bigger epsilon means a smaller noise scale. More training rows also
        # reduces the amount of noise added to each coefficient.
        feature_noise_scale = y_range / (np.maximum(feature_epsilons, 1e-12) * len(clipped_y))

        # The intercept gets its own slice of the privacy budget.
        intercept_epsilon = max(self.epsilon * self.intercept_epsilon_fraction, 1e-12)
        intercept_noise_scale = y_range / (intercept_epsilon * len(clipped_y))

        # Add Laplace noise to every coefficient. This is what makes the model
        # differentially-private-style instead of a plain Ridge regression.
        self.coef_ = coef + rng.laplace(
            loc=0.0,
            scale=feature_noise_scale,
            size=coef.shape,
        )

        # Add Laplace noise to the intercept too.
        self.intercept_ = self.model_.intercept_ + rng.laplace(
            loc=0.0,
            scale=intercept_noise_scale,
        )

        # Store these values so we can report them in the metrics table later.
        self.feature_epsilons_ = feature_epsilons
        self.intercept_epsilon_ = intercept_epsilon
        return self

    def predict(self, X):
        """Make predictions using the noisy coefficients."""
        return X @ self.coef_ + self.intercept_

    def _allocate_feature_epsilons(self, coef):
        """Split the feature privacy budget based on coefficient size.

        Features with larger absolute coefficients are treated as stronger
        signals, so they get more epsilon and therefore less noise.
        """
        feature_count = len(coef)

        # Keep the intercept budget separate, then adaptively divide the rest
        # across the feature coefficients.
        feature_budget = self.epsilon * (1.0 - self.intercept_epsilon_fraction)
        min_budget = feature_budget * self.min_feature_epsilon_fraction
        adaptive_budget = feature_budget - min_budget

        # Every feature starts with the same small amount of epsilon.
        min_per_feature = min_budget / feature_count

        # Absolute coefficient size is used as a simple importance score.
        magnitudes = np.abs(coef)
        total_magnitude = magnitudes.sum()

        if total_magnitude <= 0:
            # If all coefficients are zero, split the adaptive budget evenly.
            adaptive_share = np.full(feature_count, 1.0 / feature_count)
        else:
            # Otherwise, stronger coefficients get a larger share.
            adaptive_share = magnitudes / total_magnitude

        return min_per_feature + (adaptive_budget * adaptive_share)


def build_adaptive_dp_model(
    X,
    epsilon=DEFAULT_EPSILON,
    target_bounds=DEFAULT_TARGET_BOUNDS,
):
    """Build the full preprocessing-plus-model pipeline."""
    # Reuse the same preprocessing steps as the baseline model so comparisons
    # between baseline, fixed DP, and adaptive DP are fair.
    baseline_pipeline = build_model(X)
    preprocessor = baseline_pipeline.named_steps["preprocessor"]

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                AdaptiveNoisyRidge(
                    epsilon=epsilon,
                    target_bounds=target_bounds,
                    random_state=42,
                ),
            ),
        ]
    )


def train_adaptive_dp_model(
    epsilon=DEFAULT_EPSILON,
    target_bounds=DEFAULT_TARGET_BOUNDS,
    data_path=DATA_PATH,
):
    """Load the data, train the adaptive DP model, and calculate metrics."""
    df = load_data(data_path)

    # Remove rows where the target value is missing.
    df = df.dropna(subset=[TARGET_COLUMN])

    # X contains the input features. y contains the value we want to predict.
    X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    y = df[TARGET_COLUMN]

    # Use the same split each time so metric comparisons are stable.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_adaptive_dp_model(X_train, epsilon=epsilon, target_bounds=target_bounds)
    model.fit(X_train, y_train)

    # Predict on the holdout set so we measure performance on rows the model
    # did not train on.
    predictions = model.predict(X_test)

    ridge_model = model.named_steps["model"]

    # These metrics summarize how close the predictions are to the true sale
    # amounts. Lower MAE/RMSE is better; higher R2 is better.
    metrics = {
        "model": "adaptive_dp_noisy_ridge_regression",
        "epsilon": epsilon,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "min_feature_epsilon": ridge_model.feature_epsilons_.min(),
        "max_feature_epsilon": ridge_model.feature_epsilons_.max(),
        "intercept_epsilon": ridge_model.intercept_epsilon_,
    }
    return model, metrics


def save_adaptive_dp_metrics(metrics, path=RESULTS_PATH):
    """Save this model's metrics into the shared metrics CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([metrics])
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)

        # Replace the old adaptive DP row instead of adding duplicates every run.
        existing = existing[existing["model"] != metrics["model"]]
        new_row = pd.concat([existing, new_row], ignore_index=True)

    new_row.to_csv(path, index=False)


if __name__ == "__main__":
    # This block runs only when you start the file directly from the terminal.
    _, adaptive_dp_metrics = train_adaptive_dp_model()
    save_adaptive_dp_metrics(adaptive_dp_metrics)

    print("Adaptive DP-style model trained")
    for name, value in adaptive_dp_metrics.items():
        print(f"{name}: {value}")
