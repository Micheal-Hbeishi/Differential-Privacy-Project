# Combined Presentation Guide

## How to frame the repository

This is one research theme explored through two complementary projects:

- **Project 1 asks:** how do privacy mechanisms and data transformations affect
  model utility and explanations in a broad image-analysis workflow?
- **Project 2 asks:** can feature-aware adaptive noise preserve more accuracy
  and interpretability than uniform DP-SGD?

Do not present Project 2 as a replacement for Project 1. Present it as a more
focused extension: the first establishes the privacy-utility-explainability
trade-off, while the second investigates an adaptive way to improve it.

## Suggested presentation order

1. Define the shared problem: privacy noise protects training records but can
   reduce accuracy and destabilize explanations.
2. Introduce Project 1 as the broad demonstration using MNIST, DP-SGD, an
   autoencoder, and SmoothGrad.
3. State the limitation of uniform DP noise: all gradient coordinates are
   perturbed similarly even though learned features are not equally important.
4. Introduce Project 2 and FADP as the targeted follow-up.
5. Explain the three FADP channel masks: `0.6`, `0.8`, and `1.0`.
6. Compare baseline, DP-SGD, and FADP using utility, explanation stability, and
   membership-inference risk.
7. End with limitations and the open privacy-proof question.

## Thirty-second summary of each project

### Project 1

I use MNIST to recreate the broad workflow from Abbasi, Mori, and Saracino. A
normal CNN provides the utility baseline; DP-SGD shows the accuracy cost of
training privacy; an autoencoder demonstrates a lossy input transformation;
and SmoothGrad shows which pixels influence a prediction. It is an educational
workflow rather than an exact reproduction of the paper's full experiments.

### Project 2

The Riya et al. paper proposes Feature-Sensitive Adaptive Differential Privacy.
Instead of distributing the same noise everywhere, it ranks final convolutional
channels using activations and class gradients. Important channels receive less
noise, with the goal of preserving accuracy and Grad-CAM explanations while
still reducing membership-inference risk.

## Paper 2 results to know

- CIFAR-10 test accuracy: 88.5% baseline, 83.5% DP-SGD, and 84.0% FADP.
- Average explanation SSIM across 100 samples: 0.42 DP-SGD versus 0.85 FADP.
- Average peak overlap: 30% DP-SGD versus 83% FADP.
- Figure 3 reports MIA AUC values of 0.61 baseline, 0.45 DP-SGD, and 0.48 FADP.
- The surrounding text says baseline AUC is 0.59 rather than 0.61; mention this
  only if discussing paper quality or reproducibility.
- The selected mask configuration is `m_high=0.6`, `m_moderate=0.8`, and
  `m_low=1.0`.
- The paper reports `epsilon` from 0.1 to 3, `delta=1e-5`, clipping norm 1.0,
  batch size 64, and Adam learning rate 0.001.

Do not compare this repository's small-model values directly to the papers'
published values. Compare the ordering and gaps among methods.

## Questions your RA may ask

### Why are there two projects?

They operate at different levels. Project 1 demonstrates the overall trade-off
and several privacy/explanation tools. Project 2 isolates the noise-allocation
question and evaluates whether an adaptive mechanism improves over uniform
DP-SGD.

### What makes DP-SGD private?

Per-example clipping bounds the maximum influence of one record. Calibrated
Gaussian noise then hides that record's contribution, and a privacy accountant
tracks composition over training steps. Noise without sensitivity control and
accounting is not automatically differential privacy.

### Why does FADP use the final convolutional layer?

Its channels contain high-level spatial features and are the usual target of
Grad-CAM. That creates a direct link between channel importance, adaptive noise,
and the explanations being evaluated.

### Is FADP formally as private as ordinary DP-SGD?

That is not established by this replication. The paper gives a short guarantee,
but does not fully specify how data-dependent masks, feature sensitivity, and
multi-step composition are handled. Giving important coordinates only 60% of
the base noise means the ordinary DP-SGD epsilon cannot simply be reused.

### What does membership-inference AUC mean?

It measures whether an attacker can distinguish training members from
non-members. An AUC near 0.5 is chance. An AUC below 0.5 is not automatically
better privacy because the attacker may invert the score.

### Do stable heatmaps prove correct explanations?

No. SSIM and peak overlap measure similarity to the non-private model's
Grad-CAM map. They do not prove causal faithfulness or human usefulness.

### Why do the two projects use different explanation methods?

They follow their respective project goals. SmoothGrad is a simple pixel-level
saliency demonstration for MNIST. Grad-CAM operates on convolutional features
and directly supports the FADP paper's feature-importance mechanism.

## Claims to avoid

- Do not say either project exactly reproduces its paper.
- Do not claim the autoencoder itself provides differential privacy.
- Do not treat MIA results as a formal DP guarantee.
- Do not claim Grad-CAM or SmoothGrad reveals the model's true reasoning.
- Do not claim FADP and DP-SGD have the same epsilon without a complete proof.
- Do not hide the architecture, dataset-size, or attack simplifications.

## Before meeting your RA

1. Run both scripts and inspect both result directories.
2. Record runtime, hardware, package versions, and random seed.
3. For each project, prepare one table and one explanation image.
4. Be ready to explain why Project 2 is a focused extension of Project 1.
5. Check whether your FADP run actually improves over DP-SGD; report a negative
   result honestly if it does not.
6. Lead with the scientific question, then the methods, evidence, and caveats.

## Strong closing statement

> The first project demonstrates the general privacy-utility-explainability
> tension. The second tests a specific response: allocate noise according to
> learned feature importance. The empirical idea is promising, but the next
> research step is a reproducible large-scale evaluation and a rigorous privacy
> analysis of the data-dependent adaptive masks.
