"""Paper-aligned privacy/utility/explainability experiment.

This is a compact educational replication of Abbasi, Mori, and Saracino
(IEEE TDSC, 2025). It uses sklearn's 8x8 digits dataset so the experiment is
fully reproducible without downloading data. It is not an exact reproduction
of the paper's TensorFlow/CNN experiments on MNIST, FER, and CIFAR-10.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_aligned"
SEED = 42


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class TinyMLP:
    input_dim: int
    hidden_dim: int = 32
    classes: int = 10
    seed: int = SEED

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0, np.sqrt(2 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.normal(0, np.sqrt(2 / self.hidden_dim), (self.hidden_dim, self.classes))
        self.b2 = np.zeros(self.classes)

    def forward(self, x: np.ndarray):
        pre = x @ self.w1 + self.b1
        hidden = np.maximum(pre, 0)
        return pre, hidden, softmax(hidden @ self.w2 + self.b2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2].argmax(axis=1)

    def input_gradients(self, x: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        pre, _, _ = self.forward(x)
        if labels is None:
            labels = self.predict(x)
        selected = self.w2[:, labels].T
        return (selected * (pre > 0)) @ self.w1.T

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 35,
        batch_size: int = 128,
        learning_rate: float = 0.08,
        clip_norm: float | None = None,
        noise_multiplier: float = 0.0,
    ) -> None:
        """Train normally or with per-example clipped, Gaussian-noised gradients."""
        rng = np.random.default_rng(self.seed)
        eye = np.eye(self.classes)
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x[idx], y[idx]
                pre, hidden, probs = self.forward(xb)
                dz2 = probs - eye[yb]
                # Per-example gradients. Shapes start with the batch dimension.
                gw2 = hidden[:, :, None] * dz2[:, None, :]
                gb2 = dz2
                dh = dz2 @ self.w2.T
                dz1 = dh * (pre > 0)
                gw1 = xb[:, :, None] * dz1[:, None, :]
                gb1 = dz1
                grads = [gw1, gb1, gw2, gb2]

                if clip_norm is not None:
                    squared = sum((g.reshape(len(xb), -1) ** 2).sum(axis=1) for g in grads)
                    factors = np.minimum(1.0, clip_norm / (np.sqrt(squared) + 1e-12))
                    grads = [g * factors.reshape((-1,) + (1,) * (g.ndim - 1)) for g in grads]

                averaged = [g.mean(axis=0) for g in grads]
                if clip_norm is not None and noise_multiplier > 0:
                    std = noise_multiplier * clip_norm / len(xb)
                    averaged = [g + rng.normal(0, std, g.shape) for g in averaged]

                for parameter, gradient in zip(
                    [self.w1, self.b1, self.w2, self.b2], averaged
                ):
                    parameter -= learning_rate * gradient


def privacy_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """Conservative moments-accountant-style approximation used for comparison.

    The original paper delegates accounting to TensorFlow Privacy. This compact
    implementation uses a documented analytical upper-bound approximation, not
    a certified production accountant.
    """
    if noise_multiplier <= 0:
        return float("inf")
    return (
        sample_rate * np.sqrt(2 * steps * np.log(1 / delta)) / noise_multiplier
        + steps * sample_rate**2 / noise_multiplier**2
    )


def smoothgrad(model: TinyMLP, x: np.ndarray, noise: float, samples: int = 20) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    labels = model.predict(x)
    gradients = np.zeros_like(x)
    for _ in range(samples):
        perturbed = np.clip(x + rng.normal(0, noise, x.shape), 0, 1)
        gradients += np.abs(model.input_gradients(perturbed, labels))
    return gradients / samples


def explanation_stability(reference: np.ndarray, candidate: np.ndarray) -> float:
    a, b = reference.ravel(), candidate.ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.clip((np.corrcoef(a, b)[0, 1] + 1) / 2, 0, 1))


def reconstruct_with_autoencoder(x_train, x_test, code_size: int):
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(code_size,), activation="logistic", solver="adam",
        max_iter=160, random_state=SEED, early_stopping=True,
    )
    autoencoder.fit(x_train, x_train)
    return np.clip(autoencoder.predict(x_test), 0, 1)


def run(quick: bool = False) -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    x = digits.data.astype(float) / 16.0
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    epochs = 18 if quick else 35
    batch_size = 128
    baseline = TinyMLP(x.shape[1])
    baseline.train(x_train, y_train, epochs=epochs, batch_size=batch_size)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(x_test))
    explain_x = x_test[:80]
    reference_map = smoothgrad(baseline, explain_x, 0.0)

    dp_noises = [0.0, 0.75, 1.25, 2.0]
    explanation_noises = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    steps = epochs * int(np.ceil(len(x_train) / batch_size))
    delta = 1 / len(x_train) ** 1.1
    for dp_noise in dp_noises:
        model = TinyMLP(x.shape[1])
        model.train(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            clip_norm=1.0, noise_multiplier=dp_noise,
        )
        accuracy = accuracy_score(y_test, model.predict(x_test))
        epsilon = privacy_epsilon(dp_noise, batch_size / len(x_train), steps, delta)
        privacy_gain = 0.0 if not np.isfinite(epsilon) else 1 / epsilon
        utility_loss = max(0.0, baseline_accuracy - accuracy)
        for explanation_noise in explanation_noises:
            candidate = smoothgrad(model, explain_x, explanation_noise)
            stability = explanation_stability(reference_map, candidate)
            # Paper-inspired controllable gain, bounded by empirical stability and utility.
            explainability_gain = explanation_noise * stability * (1 - utility_loss)
            tradeoff = (privacy_gain + explainability_gain) / (2 + utility_loss)
            rows.append({
                "mechanism": "DP-SGD", "privacy_degree": dp_noise,
                "explainability_degree": explanation_noise, "epsilon": epsilon,
                "delta": delta, "accuracy": accuracy, "utility_loss": utility_loss,
                "privacy_gain": privacy_gain, "explanation_stability": stability,
                "explainability_gain": explainability_gain, "tradeoff_score": tradeoff,
            })

    # Autoencoder branch: reconstructed images are classified by the same baseline model.
    for code_size in ([8, 32] if quick else [4, 8, 16, 32]):
        reconstructed = reconstruct_with_autoencoder(x_train, x_test, code_size)
        accuracy = accuracy_score(y_test, baseline.predict(reconstructed))
        utility_loss = float(np.mean(np.abs(x_test - reconstructed)))
        privacy_gain = 1 - code_size / x.shape[1]
        for explanation_noise in explanation_noises:
            candidate = smoothgrad(baseline, reconstructed[:80], explanation_noise)
            stability = explanation_stability(reference_map, candidate)
            explainability_gain = explanation_noise * stability * (1 - utility_loss)
            tradeoff = (privacy_gain + explainability_gain) / (2 + utility_loss)
            rows.append({
                "mechanism": "Autoencoder", "privacy_degree": code_size / x.shape[1],
                "explainability_degree": explanation_noise, "epsilon": np.nan,
                "delta": np.nan, "accuracy": accuracy, "utility_loss": utility_loss,
                "privacy_gain": privacy_gain, "explanation_stability": stability,
                "explainability_gain": explainability_gain, "tradeoff_score": tradeoff,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "compatibility_matrix.csv", index=False)
    best = frame.loc[frame.groupby("mechanism")["tradeoff_score"].idxmax()]
    best.to_csv(RESULTS / "best_configurations.csv", index=False)
    metadata = {
        "dataset": "sklearn digits (8x8 grayscale)", "train_rows": len(x_train),
        "test_rows": len(x_test), "baseline_accuracy": baseline_accuracy,
        "epochs": epochs, "batch_size": batch_size, "delta": delta,
        "seed": SEED,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    make_figures(frame, baseline_accuracy)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(best[["mechanism", "privacy_degree", "explainability_degree", "accuracy", "tradeoff_score"]].to_string(index=False))
    return frame


def make_figures(frame: pd.DataFrame, baseline_accuracy: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    dp = frame[frame.mechanism == "DP-SGD"].drop_duplicates("privacy_degree")
    axes[0].plot(dp.privacy_degree, dp.accuracy, marker="o", label="DP-SGD")
    axes[0].axhline(baseline_accuracy, color="black", linestyle="--", label="Baseline")
    axes[0].set(xlabel="DP noise multiplier", ylabel="Test accuracy", title="Privacy-utility relationship")
    axes[0].legend()
    for mechanism, subset in frame.groupby("mechanism"):
        best_by_x = subset.groupby("explainability_degree").tradeoff_score.max()
        axes[1].plot(best_by_x.index, best_by_x.values, marker="o", label=mechanism)
    axes[1].set(xlabel="SmoothGrad noise", ylabel="Best trade-off score", title="Privacy-utility-explainability trade-off")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "paper_aligned_results.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short validation run")
    run(quick=parser.parse_args().quick)
