# The same research question under two resource budgets

**This is what we obtained with low resources; this is how we test the same
idea with substantially more resources.** The larger experiment is an
implemented protocol awaiting execution. More compute may improve learning,
but it does not guarantee that FADP beats DP-SGD.

The shared question is how privacy mechanisms change predictive utility and
explanations. The Abbasi–Mori–Saracino project introduces that question through
MNIST, DP-SGD, compression, and SmoothGrad. The Riya et al. project focuses it
into a comparison of baseline training, uniform noise, and feature-sensitive
adaptive noise. Each has a different purpose; do not equate their accuracies.

## Low resources versus substantial resources

| Experimental choice | Existing low-resource version | New higher-resource FADP version |
|---|---|---|
| Training data | 5,000 CIFAR-10 images | 45,000 images; 5,000 held out for validation |
| Test data | 1,000 images; validation also drawn from official test data | All 10,000 official test images; validation comes from training data |
| Architecture | Small CNN, random initialization | ImageNet-pretrained MobileNet v1 plus explicit 128-channel projection/head |
| Trainable layers | Entire small CNN | Projection/head by default; optional MobileNet fine-tuning with frozen BatchNorm |
| Training duration | 3 epochs | 30 epochs, configurable |
| Repetitions | One seed | Seeds 42, 43, 44 |
| Noise settings | Multiplier 1.0 | Multipliers 0.5, 1.0, 2.0 |
| Explanation sample | 25 images | 100 images |
| Explanations | Grad-CAM SSIM and top-20% overlap | Same definitions, to retain comparability |
| Attack | Maximum-confidence MIA | Same audit; stronger shadow-model attacks remain future work |
| Reporting | One run | Per-run files, mean/SD/count across seeds, hardware/software and runtime |
| Compute | Laptop demonstration | GPU workstation/cloud recommended; actual cost/runtime must be measured |

The high-resource defaults train 27 models: three methods × three seeds ×
three noise settings. The baseline is retrained in each setting for a paired
comparison. About 702 updates/epoch × 30 epochs × 27 models is 568,620 model
updates, versus 702 updates across the three models in the low-resource run.
Private updates also require per-example gradients. This is a large experiment,
not an implied promise of a short laptop runtime. Gradient chunks reduce peak
memory while preserving clipping of each example and one noise draw per batch.

## Run the new version

Use the existing requirements and a compatible TensorFlow environment.

```bash
python notebook/high_resource_replication.py
```

For greater architectural fidelity, train the MobileNet convolutional weights
as well as the added head (considerably more expensive):

```bash
python notebook/high_resource_replication.py --fine-tune-backbone
```

For a small pipeline check rather than a research result:

```bash
python notebook/high_resource_replication.py --epochs 1 --train-limit 128 --validation-limit 64 --test-limit 128 --seeds 42 --noise 1.0 --explanation-samples 10
```

Outputs live in `results/high_resource_replication/seed_*/noise_*/`.
`all_performance.csv` and `performance_summary.csv` collect completed runs.
Existing completed run directories are protected against overwriting.
ImageNet weights and CIFAR-10 require downloads if they are not already cached.
`--random-init` permits a structural check without pretrained weights; it must
not be presented as the intended pretrained research experiment.

## What is closer to the paper, and what remains approximate?

The new version addresses the old dataset-size, model-capacity, training-time,
and repetition limitations. It retains the paper-inspired masks 0.6/0.8/1.0,
Adam learning rate 0.001, clipping norm 1.0, and batch size 64.
The 30-epoch schedule, noise grid, train/validation split, projection head, and
seed count are our experimental choices, not claimed paper specifications.
Frozen-backbone transfer learning differs from end-to-end fine-tuning.

Equal-third importance ranking, mapping masks to an added final convolution
and its connected dense layer, and confidence-based MIA still approximate
underspecified implementation choices. Grad-CAM currently explains each
model's own predicted class: report prediction agreement alongside similarity
before interpreting heatmap agreement as feature preservation. The larger
version inherits this metric convention to support comparison.

Neither runner's shuffled fixed batches match its Poisson RDP calculation.
Treat those epsilon values as diagnostic proxies. Adaptive masks additionally
depend on private data and do not have a complete privacy proof here.
Do not claim matched formal privacy merely because noise multipliers match.
Full-data access, more epochs, and hyperparameter selection all change the
privacy analysis; published metrics also need consideration in a private release.

## Evidence and presentation

The existing presentation records the low-resource CIFAR-10 test accuracies as
26.2% baseline, 12.5% DP-SGD, and 9.0% FADP. Explanation SSIM was 0.108 for
DP-SGD and 0.273 for FADP. These are prior local measurements, not outcomes of
the new version. Consult the saved run files when presenting them.

Use this wording:

> With low resources, I demonstrated the workflow but the private models were
> near chance. FADP preserved more heatmap structure in that run without
> improving accuracy. With substantially more resources, I now use pretrained
> MobileNet features, most of CIFAR-10 for training, longer schedules, and
> repeated noise settings and seeds. The larger experiment tests whether the
> original negative result reflects undertraining or a limitation of the
> adaptive-noise implementation. Its results remain pending until it is run.

After execution, compare within-budget baseline-to-private accuracy gaps,
SSIM/overlap, attack AUC, and runtime. Report seed variability, not just the
best seed. Use validation for model choices and keep the test set for final
reporting. A noise sweep is a sensitivity study, not automatically a fair
comparison at equal certified privacy. Do not transplant paper results into
an empty high-resource results column.

## Research and implementation resources

- [Abbasi, Mori and Saracino: publication record](https://www.iris.sssup.it/handle/11382/567812).
  Establishes the broad privacy–utility–explainability framing; its full text
  is access restricted at that repository. We do not claim to reproduce
  unverified full experimental details of this paper.
- [Riya et al.: ORNL publication record](https://impact.ornl.gov/en/publications/balancing-trade-offs-adaptive-differential-privacy-in-interpretab/)
  and the supplied `references/adaptive_differential_privacy_interpretable_ml.pdf`.
  The focused adaptive-noise research anchor.
- [Abadi et al., Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133).
  Foundation for per-example clipping, noisy optimization, and composition.
- [Official MobileNet documentation](https://keras.io/api/applications/mobilenet/).
  Implementation reference for pretrained weights and preprocessing to [-1, 1].

Additional resources strengthen the protocol and explain its assumptions;
they do not establish an untested improvement or supply a missing FADP proof.
