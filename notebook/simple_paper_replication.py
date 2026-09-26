"""My small MNIST recreation of privacy, utility, and explainability ideas.

I compare a normal image classifier, a DP-SGD classifier, and an autoencoder.
I also create a SmoothGrad saliency map so I can see the explanation, not only
the final accuracy number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve all output paths from the project folder, not the terminal folder.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "simple_paper_replication"
SEED = 42

# Keep the project fast enough for a laptop while still using real MNIST images.
TRAIN_LIMIT = 5_000
TEST_LIMIT = 1_000
BATCH_SIZE = 128
EPOCHS = 5
CLIP_NORM = 1.0
DELTA = 1e-5


def dependencies():
    """Import TensorFlow Privacy only when this experiment is run."""
    try:
        import tensorflow as tf
        from dp_accounting import dp_event, rdp
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
            DPKerasAdamOptimizer,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the project dependencies first: `python -m pip install -r requirements.txt`."
        ) from error
    return tf, dp_event, rdp, DPKerasAdamOptimizer


def load_data():
    """Download MNIST once, normalize its pixels, and select a small subset."""
    tf, _, _, _ = dependencies()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Neural networks learn more reliably from small float values than 0-255 integers.
    x_train = x_train[:TRAIN_LIMIT].astype("float32") / 255.0
    y_train = y_train[:TRAIN_LIMIT]
    x_test = x_test[:TEST_LIMIT].astype("float32") / 255.0
    y_test = y_test[:TEST_LIMIT]

    # Add a channel dimension because Keras convolution layers expect (height, width, channels).
    return x_train[..., None], y_train, x_test[..., None], y_test


def classifier(tf):
    """Create one small CNN used for the normal and DP image classifiers."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10),  # Raw scores for digits 0 through 9.
        ]
    )


def train_baseline(tf, x_train, y_train):
    """Train an ordinary classifier to establish the best-utility reference."""
    model = classifier(tf)
    model.compile(
        # Legacy Adam is substantially faster with TensorFlow 2.11-2.15 on
        # Apple Silicon and is the implementation recommended by TensorFlow.
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    return model


def train_private(tf, optimizer_class, x_train, y_train, noise_multiplier):
    """Train the same classifier with per-example clipping and Gaussian DP noise."""
    model = classifier(tf)
    model.compile(
        optimizer=optimizer_class(
            l2_norm_clip=CLIP_NORM,
            noise_multiplier=noise_multiplier,
            num_microbatches=None,
            learning_rate=0.001,
        ),
        # DP-SGD must receive one loss per image so it can clip each image's gradient.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        ),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    return model


def epsilon(dp_event, rdp, noise_multiplier):
    """Compute the DP epsilon reported for the chosen training configuration."""
    steps = EPOCHS * int(np.ceil(TRAIN_LIMIT / BATCH_SIZE))
    sampling_rate = BATCH_SIZE / TRAIN_LIMIT
    accountant = rdp.RdpAccountant(orders=np.arange(1.1, 64.0, 0.1))
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_rate, dp_event.GaussianDpEvent(noise_multiplier)
        ),
        count=steps,
    )
    return float(accountant.get_epsilon(DELTA))


def train_autoencoder(tf, x_train):
    """Train a small bottleneck model that reconstructs a privacy-reduced image."""
    inputs = tf.keras.Input(shape=(28, 28, 1))
    encoded = tf.keras.layers.Flatten()(inputs)
    encoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(28 * 28, activation="sigmoid")(encoded)
    outputs = tf.keras.layers.Reshape((28, 28, 1))(decoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="mae",
    )
    model.fit(x_train, x_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    return model


def smoothgrad(tf, model, image, noise_level=0.2, samples=20):
    """Average gradients from noisy copies of an image to create a saliency map."""
    # Predict the class once; each noisy copy explains that same chosen class.
    label = int(tf.argmax(model(image[None, ...], training=False)[0]))
    total = np.zeros_like(image)
    rng = np.random.default_rng(SEED)
    for _ in range(samples):
        noisy = np.clip(image + rng.normal(0, noise_level, image.shape), 0, 1)
        tensor = tf.convert_to_tensor(noisy[None, ...], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            score = model(tensor, training=False)[0, label]
        total += np.abs(tape.gradient(score, tensor).numpy()[0])
    return total / samples, label


def accuracy(tf, model, x_test, y_test):
    """Return a simple decimal accuracy, such as 0.91 for 91%."""
    predictions = tf.argmax(model(x_test, training=False), axis=1).numpy()
    return float(np.mean(predictions == y_test))


def save_saliency_image(saliency, image, label):
    """Save the original digit and its SmoothGrad explanation side by side."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title(f"MNIST digit: {label}")
    axes[1].imshow(saliency.squeeze(), cmap="magma")
    axes[1].set_title("SmoothGrad saliency")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "smoothgrad_example.png", dpi=180)
    plt.close(figure)


def run():
    """Run the simple paper-inspired comparison and save its outputs."""
    tf, dp_event, rdp, optimizer_class = dependencies()
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data()

    baseline = train_baseline(tf, x_train, y_train)
    private = train_private(tf, optimizer_class, x_train, y_train, noise_multiplier=1.0)
    autoencoder = train_autoencoder(tf, x_train)
    reconstructed = autoencoder(x_test, training=False)

    baseline_accuracy = accuracy(tf, baseline, x_test, y_test)
    private_accuracy = accuracy(tf, private, x_test, y_test)
    reconstructed_accuracy = accuracy(tf, baseline, reconstructed, y_test)
    private_epsilon = epsilon(dp_event, rdp, noise_multiplier=1.0)
    reconstruction_error = float(tf.reduce_mean(tf.abs(x_test - reconstructed)))

    # These rows express the paper's central trade-off: privacy versus useful predictions.
    rows = [
        {"approach": "Baseline", "accuracy": baseline_accuracy, "privacy_measure": "none", "utility_loss": 0.0},
        {
            "approach": "DP-SGD", "accuracy": private_accuracy,
            "privacy_measure": f"epsilon={private_epsilon:.2f}, delta={DELTA}",
            "utility_loss": baseline_accuracy - private_accuracy,
        },
        {
            "approach": "Autoencoder", "accuracy": reconstructed_accuracy,
            "privacy_measure": "32-value image bottleneck",
            "utility_loss": reconstruction_error,
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    saliency, label = smoothgrad(tf, baseline, x_test[0])
    save_saliency_image(saliency, x_test[0], label)
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps({"train_images": TRAIN_LIMIT, "test_images": TEST_LIMIT, "seed": SEED}, indent=2)
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
