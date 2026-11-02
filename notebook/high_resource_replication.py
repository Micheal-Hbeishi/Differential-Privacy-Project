"""Higher-resource CIFAR-10 experiment; see RESOURCE_COMPARISON.md.

Reuses the transparent low-resource FADP mechanism and evaluation. This is
paper-aligned transfer learning, not an exact reproduction or a FADP DP proof.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import fadp_replication as low


@dataclass(frozen=True)
class Config(low.Config):
    train_limit: int = 45_000
    validation_limit: int = 5_000
    test_limit: int = 10_000
    epochs: int = 30
    explanation_samples: int = 100


def load_data(tf, cfg):
    (x, y), (test, test_y) = tf.keras.datasets.cifar10.load_data()
    order = np.random.default_rng(cfg.seed).permutation(len(x))
    validation = order[:cfg.validation_limit]
    train = order[cfg.validation_limit:cfg.validation_limit + cfg.train_limit]
    if len(train) != cfg.train_limit or cfg.test_limit > len(test):
        raise ValueError("Requested split exceeds CIFAR-10 capacity")
    normalize = lambda a: a.astype('float32') / 255.0
    return (normalize(x[train]), y[train, 0].astype('int32'),
            normalize(x[validation]), y[validation, 0].astype('int32'),
            normalize(test[:cfg.test_limit]), test_y[:cfg.test_limit, 0].astype('int32'))


def model_builder(tf, pretrained=True, fine_tune=False):
    inputs = tf.keras.Input((32, 32, 3), name='image')
    x = tf.keras.layers.Resizing(224, 224)(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)
    backbone = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3), include_top=False,
        weights='imagenet' if pretrained else None)
    backbone.trainable = fine_tune
    # Batch-dependent normalization would invalidate independent-example
    # gradient clipping. Keep pretrained BN statistics fixed in every method.
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    x = backbone(x, training=False)
    # Explicit projection gives the existing channel-to-parameter mask mapping
    # a well-defined target. This added layer is a replication choice.
    x = tf.keras.layers.Conv2D(128, 1, activation='relu', name='last_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='feature_dense')(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(10, name='logits')(x))


def chunked_gradients(tf, model, images, labels, chunk_size):
    # Clip EACH example over ALL trainable parameters before aggregating.
    # Noise is added by the original trainer once per logical batch.
    total = [tf.zeros_like(v) for v in model.trainable_variables]
    count = int(tf.shape(images)[0])
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        grads = ORIGINAL_GRADIENTS(tf, model, images[start:end], labels[start:end])
        mean = low.clipped_average(tf, grads, ACTIVE_CLIP)
        total = [a + b * (end - start) for a, b in zip(total, mean)]
    # Original trainer clips again: the mean of clipped vectors is already
    # inside the clipping ball, so this preserves the aggregate.
    return tuple(tf.expand_dims(g / count, 0) for g in total)


ORIGINAL_GRADIENTS = low.per_example_gradients
ACTIVE_CLIP = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-limit', type=int, default=45_000)
    parser.add_argument('--validation-limit', type=int, default=5_000)
    parser.add_argument('--test-limit', type=int, default=10_000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--noise', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--gradient-chunk', type=int, default=4)
    parser.add_argument('--fine-tune-backbone', action='store_true')
    parser.add_argument('--random-init', action='store_true', help='Offline structural check; not the research configuration')
    parser.add_argument('--explanation-samples', type=int, default=100)
    args = parser.parse_args()
    if min(args.epochs, args.train_limit, args.validation_limit, args.test_limit,
           args.gradient_chunk, args.explanation_samples) <= 0 or min(args.noise) <= 0:
        parser.error('Sizes, epochs and noise must be positive')
    if args.train_limit < 64 or args.train_limit + args.validation_limit > 50_000 or args.test_limit > 10_000:
        parser.error('Need >=64 training examples and valid CIFAR-10 split sizes')
    if args.explanation_samples > args.test_limit:
        parser.error('Explanation sample count exceeds test split')
    low.load_cifar10 = load_data
    low.build_model = lambda tf: model_builder(tf, not args.random_init, args.fine_tune_backbone)
    low.per_example_gradients = lambda tf, m, x, y: chunked_gradients(tf, m, x, y, args.gradient_chunk)
    root = low.ROOT / 'results' / 'high_resource_replication'
    rows = []
    for seed in args.seeds:
        for noise in args.noise:
            tf, _, _ = low.dependencies()
            tf.keras.backend.clear_session()
            cfg = Config(seed=seed, noise_multiplier=noise, epochs=args.epochs,
                         train_limit=args.train_limit, validation_limit=args.validation_limit,
                         test_limit=args.test_limit, explanation_samples=args.explanation_samples)
            low.OUTPUT_DIR = root / f'seed_{seed}' / f'noise_{noise:g}'
            if (low.OUTPUT_DIR / 'run_metadata.json').exists():
                raise FileExistsError(f'Existing run: {low.OUTPUT_DIR}; move it before rerunning')
            start = time.perf_counter()
            low.run(cfg)
            metadata_path = low.OUTPUT_DIR / 'run_metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['resource_tier'] = 'high'
            metadata['architecture'] = {'backbone': 'MobileNet v1', 'imagenet_weights': not args.random_init,
                'fine_tune_backbone': args.fine_tune_backbone, 'batch_normalization': 'frozen',
                'input_resolution': 224, 'added_projection_channels': 128}
            metadata['runtime_seconds'] = time.perf_counter() - start
            metadata['environment'] = {'python': platform.python_version(), 'tensorflow': tf.__version__,
                'platform': platform.platform(), 'devices': [str(d) for d in tf.config.list_physical_devices()]}
            metadata['limitations'] = [
                'MobileNet plus a new projection/head; frozen backbone by default, optional fine-tuning.',
                'Equal-third channel ranking and mask mapping remain approximations of FADP.',
                'Confidence MIA remains a lightweight audit, not a shadow-model attack.',
                'Shuffled fixed batches do not match the Poisson accountant: both epsilons are proxies.',
                'Hyperparameter search and evaluation releases have no composed privacy guarantee.']
            metadata['privacy_accounting']['note'] = 'Both values are diagnostic proxies: training uses shuffled fixed batches, not Poisson sampling. FADP additionally uses unaccounted data-dependent masks. Neither value is a certified guarantee for this run.'
            metadata_path.write_text(json.dumps(metadata, indent=2))
            frame = pd.read_csv(low.OUTPUT_DIR / 'performance.csv')
            frame['seed'], frame['noise_multiplier'] = seed, noise
            rows.append(frame)
            combined = pd.concat(rows, ignore_index=True)
            combined.to_csv(root / 'all_performance.csv', index=False)
            combined.groupby(['noise_multiplier', 'model', 'split'])[['accuracy', 'loss']].agg(['mean', 'std', 'count']).to_csv(root / 'performance_summary.csv')


if __name__ == '__main__':
    main()
