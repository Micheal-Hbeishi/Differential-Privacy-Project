"""Small, auditable replication of the FADP paper on CIFAR-10.

The experiment compares a non-private CNN, uniform DP-SGD, and a practical
Feature-Sensitive Adaptive Differential Privacy (FADP) approximation.  It is
designed for a laptop and intentionally does not claim to reproduce the
paper's MobileNet-scale numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "fadp_replication"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    train_limit: int = 5_000
    validation_limit: int = 1_000
    test_limit: int = 1_000
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    delta: float = 1e-5
    high_mask: float = 0.6
    moderate_mask: float = 0.8
    low_mask: float = 1.0
    explanation_samples: int = 25
    peak_fraction: float = 0.20


def dependencies():
    try:
        import tensorflow as tf
        from dp_accounting import dp_event, rdp
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies first: python -m pip install -r requirements.txt"
        ) from error
    return tf, dp_event, rdp


def load_cifar10(tf, cfg):
    """Load disjoint, normalized CIFAR-10 train/validation/test subsets."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train[:, 0], y_test[:, 0]
    x_train = x_train[: cfg.train_limit].astype("float32") / 255.0
    y_train = y_train[: cfg.train_limit]
    x_val = x_test[: cfg.validation_limit].astype("float32") / 255.0
    y_val = y_test[: cfg.validation_limit]
    start = cfg.validation_limit
    stop = start + cfg.test_limit
    x_eval = x_test[start:stop].astype("float32") / 255.0
    y_eval = y_test[start:stop]
    return x_train, y_train, x_val, y_val, x_eval, y_eval


def build_model(tf):
    """Compact CNN with a named final convolutional layer for Grad-CAM/FADP."""
    inputs = tf.keras.Input((32, 32, 3), name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu", name="last_conv"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="channel_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="feature_dense")(x)
    outputs = tf.keras.layers.Dense(10, name="logits")(x)
    return tf.keras.Model(inputs, outputs)


def make_dataset(tf, x, y, cfg, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=cfg.seed, reshuffle_each_iteration=True)
    return ds.batch(cfg.batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)


def train_baseline(tf, model, train_ds, cfg):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(train_ds, epochs=cfg.epochs, verbose=2)


def channel_masks(tf, model, images, labels, cfg):
    """Create three channel tiers from gradient-weighted final-layer activations."""
    probe = tf.keras.Model(
        model.inputs, [model.get_layer("last_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        activations, logits = probe(images, training=True)
        tape.watch(activations)
        chosen_scores = tf.gather(logits, labels, axis=1, batch_dims=1)
        score = tf.reduce_sum(chosen_scores)
    gradients = tape.gradient(score, activations)
    importance = tf.reduce_mean(
        tf.abs(gradients * activations), axis=(0, 1, 2)
    )
    order = tf.argsort(importance)
    count = tf.shape(order)[0]
    first_cut, second_cut = count // 3, (2 * count) // 3
    masks = tf.ones_like(importance) * cfg.low_mask
    masks = tf.tensor_scatter_nd_update(
        masks, tf.expand_dims(order[first_cut:second_cut], 1),
        tf.ones(second_cut - first_cut) * cfg.moderate_mask,
    )
    masks = tf.tensor_scatter_nd_update(
        masks, tf.expand_dims(order[second_cut:], 1),
        tf.ones(count - second_cut) * cfg.high_mask,
    )
    return tf.stop_gradient(masks), tf.stop_gradient(importance)


def mask_for_variable(tf, variable, channel_mask):
    """Map final-convolution channel importance onto connected parameters."""
    # Keras 3 shortened ``variable.name`` to values such as ``kernel``.  The
    # path retains the owning layer and also works on older tf.keras releases.
    name = getattr(variable, "path", variable.name)
    if "last_conv" in name and "kernel" in name:
        return tf.reshape(channel_mask, (1, 1, 1, -1))
    if "last_conv" in name and "bias" in name:
        return channel_mask
    # GlobalAveragePooling makes the next dense layer's input rows correspond
    # one-to-one with last-convolution channels.
    if "feature_dense" in name and "kernel" in name:
        return tf.reshape(channel_mask, (-1, 1))
    return tf.ones_like(variable)


def per_example_gradients(tf, model, images, labels):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def gradient_for_one(sample):
        image, label = sample
        with tf.GradientTape() as tape:
            logits = model(image[None, ...], training=True)
            loss = loss_fn(label[None], logits)[0]
        return tuple(tape.gradient(loss, model.trainable_variables))

    return tf.vectorized_map(gradient_for_one, (images, labels))


def clipped_average(tf, per_grads, clip_norm):
    squared_norms = None
    for grad in per_grads:
        axes = tuple(range(1, len(grad.shape)))
        term = tf.reduce_sum(tf.square(grad), axis=axes)
        squared_norms = term if squared_norms is None else squared_norms + term
    factors = tf.minimum(1.0, clip_norm / (tf.sqrt(squared_norms) + 1e-12))
    averaged = []
    for grad in per_grads:
        shape = (-1,) + (1,) * (len(grad.shape) - 1)
        averaged.append(tf.reduce_mean(grad * tf.reshape(factors, shape), axis=0))
    return averaged


def train_private(tf, model, train_ds, cfg, adaptive):
    """Train with per-example clipping and uniform or feature-adaptive noise."""
    optimizer = tf.keras.optimizers.Adam(cfg.learning_rate)
    mask_history = []
    for epoch in range(cfg.epochs):
        losses = []
        for images, labels in train_ds:
            if adaptive:
                feature_mask, _ = channel_masks(tf, model, images, labels, cfg)
                mask_history.append(feature_mask.numpy())
            else:
                feature_mask = tf.ones(model.get_layer("last_conv").filters)
            per_grads = per_example_gradients(tf, model, images, labels)
            gradients = clipped_average(tf, per_grads, cfg.clip_norm)
            noisy_gradients = []
            stddev = cfg.noise_multiplier * cfg.clip_norm / cfg.batch_size
            for gradient, variable in zip(gradients, model.trainable_variables):
                noise_mask = mask_for_variable(tf, variable, feature_mask)
                noise = tf.random.normal(tf.shape(gradient), stddev=stddev)
                noisy_gradients.append(gradient + noise * noise_mask)
            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))
            logits = model(images, training=False)
            losses.append(
                float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )))
            )
        label = "FADP" if adaptive else "DP-SGD"
        print(f"{label} epoch {epoch + 1}/{cfg.epochs} - loss: {np.mean(losses):.4f}")
    return np.asarray(mask_history)


def accuracy_and_loss(tf, model, x, y, batch_size):
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True
    ).numpy()
    return float(np.mean(np.argmax(logits, axis=1) == y)), float(np.mean(loss))


def privacy_epsilon(dp_event, rdp, cfg, effective_multiplier):
    """Illustrative RDP calculation using the weakest-noise coordinate.

    This is not a proof for data-dependent FADP masks; see README.md.
    """
    steps = cfg.epochs * (cfg.train_limit // cfg.batch_size)
    sampling_rate = cfg.batch_size / cfg.train_limit
    accountant = rdp.RdpAccountant(orders=np.arange(1.1, 128.0, 0.1))
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_rate, dp_event.GaussianDpEvent(effective_multiplier)
        ),
        count=steps,
    )
    return float(accountant.get_epsilon(cfg.delta))


def grad_cam(tf, model, images):
    probe = tf.keras.Model(
        model.inputs, [model.get_layer("last_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        activations, logits = probe(images, training=False)
        classes = tf.argmax(logits, axis=1, output_type=tf.int32)
        scores = tf.gather(logits, classes, axis=1, batch_dims=1)
    grads = tape.gradient(scores, activations)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    maps = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))
    maxima = tf.reduce_max(maps, axis=(1, 2), keepdims=True)
    return (maps / (maxima + 1e-12)).numpy(), classes.numpy()


def explanation_metrics(reference_maps, candidate_maps, peak_fraction):
    ssim_values, overlaps = [], []
    for reference, candidate in zip(reference_maps, candidate_maps):
        ssim_values.append(structural_similarity(reference, candidate, data_range=1.0))
        k = max(1, int(reference.size * peak_fraction))
        reference_peak = np.argpartition(reference.ravel(), -k)[-k:]
        candidate_peak = np.argpartition(candidate.ravel(), -k)[-k:]
        overlaps.append(len(np.intersect1d(reference_peak, candidate_peak)) / k)
    return float(np.mean(ssim_values)), float(np.mean(overlaps))


def membership_auc(tf, model, member_x, nonmember_x, batch_size):
    """Simple black-box MIA: maximum softmax confidence is the attack score."""
    member = tf.nn.softmax(model.predict(member_x, batch_size=batch_size, verbose=0))
    nonmember = tf.nn.softmax(model.predict(nonmember_x, batch_size=batch_size, verbose=0))
    scores = np.concatenate([np.max(member, axis=1), np.max(nonmember, axis=1)])
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    return float(roc_auc_score(labels, scores))


def save_gradcam_figure(images, maps_by_model, classes_by_model):
    names = list(maps_by_model)
    count = min(3, len(images))
    figure, axes = plt.subplots(len(names), count, figsize=(3.4 * count, 3 * len(names)))
    for row, name in enumerate(names):
        for column in range(count):
            axis = axes[row, column]
            axis.imshow(images[column])
            heatmap = np.array(
                Image.fromarray(np.uint8(maps_by_model[name][column] * 255)).resize(
                    (32, 32)
                )
            ) / 255.0
            axis.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axis.set_title(f"{name}: class {classes_by_model[name][column]}")
            axis.axis("off")
    figure.suptitle("Grad-CAM comparison (same CIFAR-10 images)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "gradcam_comparison.png", dpi=180)
    plt.close(figure)


def run(cfg):
    tf, dp_event, rdp = dependencies()
    tf.keras.utils.set_random_seed(cfg.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10(tf, cfg)
    train_ds = make_dataset(tf, x_train, y_train, cfg, training=True)

    initial = build_model(tf)
    initial_weights = initial.get_weights()
    models = {name: build_model(tf) for name in ("Baseline", "DP-SGD", "FADP")}
    for model in models.values():
        model.set_weights(initial_weights)

    train_baseline(tf, models["Baseline"], train_ds, cfg)
    train_private(tf, models["DP-SGD"], train_ds, cfg, adaptive=False)
    mask_history = train_private(tf, models["FADP"], train_ds, cfg, adaptive=True)

    dp_epsilon = privacy_epsilon(dp_event, rdp, cfg, cfg.noise_multiplier)
    fadp_epsilon = privacy_epsilon(
        dp_event, rdp, cfg, cfg.noise_multiplier * cfg.high_mask
    )
    performance_rows = []
    for name, model in models.items():
        for split, x, y in (
            ("train", x_train, y_train),
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ):
            acc, loss = accuracy_and_loss(tf, model, x, y, cfg.batch_size)
            performance_rows.append(
                {"model": name, "split": split, "accuracy": acc, "loss": loss}
            )
    pd.DataFrame(performance_rows).to_csv(OUTPUT_DIR / "performance.csv", index=False)

    sample_x = x_test[: cfg.explanation_samples]
    maps, classes = {}, {}
    for name, model in models.items():
        maps[name], classes[name] = grad_cam(tf, model, sample_x)
    explanation_rows = []
    for name in ("DP-SGD", "FADP"):
        ssim, overlap = explanation_metrics(
            maps["Baseline"], maps[name], cfg.peak_fraction
        )
        explanation_rows.append(
            {"model": name, "mean_ssim_vs_baseline": ssim,
             "mean_peak_overlap_vs_baseline": overlap}
        )
    pd.DataFrame(explanation_rows).to_csv(
        OUTPUT_DIR / "interpretability.csv", index=False
    )
    save_gradcam_figure(sample_x, maps, classes)

    mia_rows = []
    attack_count = min(len(x_train), len(x_test), 1_000)
    for name, model in models.items():
        mia_rows.append(
            {"model": name, "confidence_attack_auc": membership_auc(
                tf, model, x_train[:attack_count], x_test[:attack_count], cfg.batch_size
            )}
        )
    pd.DataFrame(mia_rows).to_csv(OUTPUT_DIR / "membership_inference.csv", index=False)

    metadata = {
        "paper": {
            "title": "Balancing Trade-offs: Adaptive Differential Privacy in Interpretable Machine Learning Models",
            "doi": "10.1109/PST65910.2025.11268818",
        },
        "config": asdict(cfg),
        "privacy_accounting": {
            "dp_sgd_epsilon": dp_epsilon,
            "fadp_illustrative_epsilon": fadp_epsilon,
            "note": "This substitutes the smallest mask into a standard accountant. It shows why reusing the DP-SGD epsilon is unsafe, but it is not a formal guarantee for data-dependent FADP masks.",
        },
        "mean_fadp_channel_mask": float(np.mean(mask_history)),
        "limitations": [
            "Small CNN and subset replace the paper's ImageNet-pretrained MobileNet and full datasets.",
            "The paper does not publish code or fully specify how channel masks map to every parameter; this implementation documents one reproducible mapping.",
            "The confidence-threshold MIA is a lightweight audit, not a shadow-model attack.",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(performance_rows).to_string(index=False))
    print(f"\nDP-SGD epsilon: {dp_epsilon:.3f}; illustrative FADP epsilon: {fadp_epsilon:.3f}")
    print(f"Saved outputs to {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--train-limit", type=int, default=Config.train_limit)
    parser.add_argument("--test-limit", type=int, default=Config.test_limit)
    args = parser.parse_args()
    return Config(epochs=args.epochs, train_limit=args.train_limit, test_limit=args.test_limit)


if __name__ == "__main__":
    run(parse_args())
