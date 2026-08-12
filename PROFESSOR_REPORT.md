# Paper-Aligned Replication Report

## Project position

This repository contains two distinct experiments:

1. a legacy tabular Ridge-regression demonstration using chocolate-sales data; and
2. a paper-aligned image-classification replication based on Abbasi, Mori, and Saracino, “Trading-Off Privacy, Utility, and Explainability in Deep Learning-Based Image Data Analysis,” *IEEE Transactions on Dependable and Secure Computing*, 2025.

The second experiment is the academically relevant comparison. It is a compact replication of the paper's methodology, not a claim of exact numerical reproduction.

## Research question

How do adjustable privacy and explanation parameters affect image-classification utility, and which configuration maximizes a combined privacy-utility-explainability score?

## Alignment with the paper

| Paper component | Replication implementation |
|---|---|
| Image classification | Stratified classification of 8x8 handwritten-digit images |
| Neural classifier | One-hidden-layer ReLU network implemented in NumPy |
| Differential privacy mechanism | Per-example gradient computation, global L2 clipping, and Gaussian noise during SGD |
| Autoencoder mechanism | Bottleneck neural autoencoder with multiple code sizes |
| SmoothGrad | Averaged input-gradient saliency under Gaussian input perturbations |
| Privacy gain | `1 / epsilon` for DP; `1 - code_size / image_size` for autoencoders |
| Utility loss | Baseline-minus-private accuracy for DP; reconstruction divergence for autoencoders |
| Explainability | Controllable SmoothGrad gain, supplemented by empirical saliency-map stability |
| Compatibility matrix | Every privacy-degree and explainability-degree configuration is saved and ranked |

## Experimental design

The experiment uses `sklearn.datasets.load_digits`, containing 1,797 grayscale images and ten classes. Pixels are normalized to `[0,1]`, and a fixed stratified 80/20 split with seed 42 is used. A non-private neural classifier establishes baseline accuracy.

The DP-SGD branch evaluates Gaussian noise multipliers `0.0`, `0.75`, `1.25`, and `2.0`, with per-example gradient clipping at norm 1.0. The autoencoder branch evaluates bottlenecks of 4, 8, 16, and 32 values compared with the 64-pixel input. Both branches evaluate SmoothGrad noise from `0.0` through `1.0`.

For every configuration, the program records privacy gain, utility loss, explanation stability, explainability gain, and the combined trade-off score. It then selects the maximum score separately for each privacy mechanism.

## Important limitations

- The original paper uses CNNs and MNIST, FER, and CIFAR-10; this replication uses a smaller built-in digits dataset to remain fast and download-free.
- The epsilon calculation is a documented moments-accountant-style analytical approximation. It is useful for comparative coursework but is not a certified production privacy accountant.
- Explanation quality is difficult to reduce to one scalar. This implementation retains the paper-inspired controllable SmoothGrad measure and adds correlation-based stability against the baseline saliency map.
- The autoencoder utility definition follows the paper's reconstruction-divergence concept but uses mean absolute pixel divergence rather than Earth Mover's Distance.
- A single dataset is used, so the compatibility matrix has privacy and explainability dimensions but not multiple stakeholder datasets.

These boundaries should be stated during presentation. They make the claims appropriately scoped and identify direct avenues for a full reproduction.

## Results from the verified run

The non-private classifier achieved **92.50% test accuracy**. Introducing per-example clipping reduced the DP-SGD branch to approximately 77% accuracy; increasing the Gaussian noise multiplier from 0.75 to 2.0 reduced the estimated epsilon from 15.66 to 4.48, while accuracy changed only slightly from 77.22% to 76.94%. This is consistent with the paper's qualitative observation that applying a privacy mechanism creates an initial utility loss that can remain comparatively stable as the privacy parameter increases.

Within the tested grid, the highest DP-SGD trade-off score was **0.4169**, at noise multiplier 2.0 and SmoothGrad noise 1.0. Its measured test accuracy was 76.94%, with estimated `(epsilon, delta) = (4.48, 0.000336)`.

The best autoencoder configuration used a 32-value bottleneck (50% of the 64-pixel input) and SmoothGrad noise 1.0. It retained **89.17% classification accuracy** and produced a trade-off score of **0.6701**. Smaller bottlenecks in this compact implementation collapsed toward chance-level classification, illustrating that maximum compression did not provide the best practical privacy-utility balance.

The two mechanism scores should be interpreted within each branch. As in the source paper, their privacy and utility terms have different definitions, so a direct numerical ranking between DP-SGD and autoencoder privacy is not a formal equivalence.

## Reproduction

```bash
python notebook/paper_aligned_replication.py
```

The run produces:

- `results/paper_aligned/compatibility_matrix.csv`
- `results/paper_aligned/best_configurations.csv`
- `results/paper_aligned/run_metadata.json`
- `results/paper_aligned/paper_aligned_results.png`

## Suggested presentation conclusion

“The implementation reproduces the paper's experimental logic—two tunable privacy mechanisms, SmoothGrad explanations, explicit utility loss, and compatibility-matrix optimization—on a smaller image-classification testbed. It demonstrates the methodology and expected trade-offs, while exact reproduction would require the original CNN architectures, three full datasets, TensorFlow Privacy accounting, and the paper's complete multi-stakeholder setup.”
