# Differential Privacy, Utility, and Explainability

This repository contains **two related but separate paper-inspired projects**.
Both investigate privacy, predictive utility, and model explanations, but they
ask different questions and use different techniques.

## Project 1: broad privacy-utility-explainability workflow

Based on:

> Wisam Abbasi, Paolo Mori, and Andrea Saracino, "Trading-Off Privacy,
> Utility, and Explainability in Deep Learning-Based Image Data Analysis,"
> *IEEE Transactions on Dependable and Secure Computing*, vol. 22, no. 1,
> pp. 388-405, 2025. DOI:
> [10.1109/TDSC.2024.3400608](https://doi.org/10.1109/TDSC.2024.3400608)

This introductory experiment uses MNIST to demonstrate four parts of the
paper's broad workflow:

- a normal CNN accuracy baseline;
- DP-SGD with per-example clipping and Gaussian noise;
- an autoencoder bottleneck as a simple data-transformation example;
- SmoothGrad saliency for prediction explanations.

Run it with:

```bash
python notebook/simple_paper_replication.py
```

Outputs are written to `results/simple_paper_replication/`.

## Project 2: feature-sensitive adaptive differential privacy

Based on:

> Farhin Farhad Riya, Shahinul Hoque, Yingyuan Yang, Jinyuan Sun, and Olivera
> Kotevska, "Balancing Trade-offs: Adaptive Differential Privacy in
> Interpretable Machine Learning Models," *2025 22nd Annual International
> Conference on Privacy, Security, and Trust (PST)*, pp. 1-6, 2025. DOI:
> [10.1109/PST65910.2025.11268818](https://doi.org/10.1109/PST65910.2025.11268818)

The attached paper is stored at
[`references/adaptive_differential_privacy_interpretable_ml.pdf`](references/adaptive_differential_privacy_interpretable_ml.pdf),
with a BibTeX entry in [`references/CITATION.bib`](references/CITATION.bib).

This experiment uses a manageable CIFAR-10 subset and compares three
identically initialized CNNs:

1. **Baseline:** ordinary, non-private training.
2. **DP-SGD:** per-example clipping and uniform Gaussian noise.
3. **FADP:** clipping plus feature-sensitive noise masks derived from
   gradient-weighted final-convolution activations.

The FADP masks use the paper's selected values:

- high-importance channels: `0.6`;
- moderate-importance channels: `0.8`;
- low-importance channels: `1.0`.

The experiment evaluates accuracy/loss, Grad-CAM SSIM, top-20% peak overlap,
a confidence-based membership-inference attack, and an illustrative privacy
calculation.

Run it with:

```bash
python notebook/fadp_replication.py
```

For a quick check:

```bash
python notebook/fadp_replication.py --epochs 1 --train-limit 256 --test-limit 128
```

Outputs are written to `results/fadp_replication/`.

## Setup

Python 3.10 or 3.11 is recommended. Both projects can use the same environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

TensorFlow 2.14-2.15 is used because Project 1 depends on TensorFlow Privacy
0.9.0. Project 2 uses explicit gradient clipping/noise and the standalone
privacy accountant, so it also runs in that environment.

## Project map

| Item | Project 1 | Project 2 |
|---|---|---|
| Main paper | Abbasi, Mori, Saracino | Riya et al. |
| Dataset | MNIST subset | CIFAR-10 subset |
| Main privacy method | Standard DP-SGD | DP-SGD versus adaptive FADP |
| Explanation method | SmoothGrad | Grad-CAM |
| Additional method | Autoencoder | Membership-inference audit |
| Main output folder | `results/simple_paper_replication/` | `results/fadp_replication/` |
| Script | `notebook/simple_paper_replication.py` | `notebook/fadp_replication.py` |

## Scope and limitations

Both experiments are **small-scale concept replications**, not exact numerical
reproductions. They deliberately use smaller models and datasets so they can
run locally.

The FADP paper does not publish enough implementation detail to reconstruct
every choice, including the clustering rule, parameter-mask mapping, MIA
implementation, and peak-overlap definition. This repository makes its choices
explicit. More importantly, its FADP epsilon is only illustrative: a formal
privacy proof must account for data-dependent masks and feature sensitivity.

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for a combined presentation guide.
