# Presentation Script: Differential Privacy, Utility, and Explainability

## Opening

Hello. My project examines a central problem in privacy-preserving machine
learning: how can we protect individual training records without losing too
much model accuracy or making the model's explanations unstable?

I explored this question through two related experiments. Project 1 presents
the broad privacy–utility–explainability trade-off using MNIST. Project 2
focuses on whether feature-sensitive adaptive noise can improve on standard
DP-SGD using CIFAR-10.

These are small-scale, paper-inspired experiments. They are not exact
reproductions of the papers' complete experimental setups.

## Project 1: broad trade-off

Project 1 uses handwritten MNIST digits and compares three approaches.

First, I trained a normal convolutional neural network, or CNN, as the
non-private baseline. Second, I trained a model using differentially private
stochastic gradient descent, known as DP-SGD. Third, I passed the images
through an autoencoder to examine the effect of lossy data transformation. I
also generated a SmoothGrad heatmap to visualize which pixels influenced a
prediction.

A CNN is an image model that learns visual patterns such as edges, shapes, and
textures. 

## How DP-SGD works

Normal training calculates gradients that tell the model how to change its
parameters to reduce prediction error. DP-SGD modifies this process in two
important ways.

First, it clips each example's gradient. This limits how strongly one training
record can influence the model. Second, it adds Gaussian noise to the clipped
gradients before updating the model. A privacy accountant then accumulates the
privacy cost across all training steps.

That cost is reported using epsilon and delta. A smaller epsilon generally
means stronger privacy. However, stronger privacy normally requires more
noise, and that noise can reduce accuracy.

## Project 1 results

The ordinary MNIST baseline achieved 91.4 percent accuracy. The DP-SGD model
achieved 62.5 percent accuracy, with epsilon 2.79 and delta 0.00001. The
autoencoder approach produced 22.7 percent downstream classification
accuracy.

These results demonstrate the expected utility cost. The private model lost
28.9 percentage points compared with the baseline. The autoencoder performed
poorly because it compressed each 784-pixel image into only 32 values and then
reconstructed an imperfect image. The classifier was trained on original
images, so these reconstructions also created a distribution shift.

The autoencoder should not be described as formally differentially private.
It demonstrates information-reducing compression, not a mathematical privacy
guarantee.

## Project 2: feature-sensitive adaptive privacy

Project 2 is based on the feature-sensitive adaptive differential privacy idea
described by Riya and colleagues. It compares three identically initialized
CNNs: a non-private baseline, standard DP-SGD, and FADP.

Standard DP-SGD applies uniform Gaussian noise. FADP instead measures the
importance of the 64 channels in the final convolutional layer. It uses each
channel's activation and class gradient to rank those channels.

The channels are divided into three groups. High-importance channels receive a
noise mask of 0.6, moderate-importance channels receive 0.8, and
low-importance channels receive 1.0. The goal is to preserve important learned
features by adding less noise to them.

## Understanding the training display

Project 2 uses 5,000 training images, a batch size of 64, and three epochs.
When the terminal displays `Epoch 1/3`, it means the model is making its first
of three complete passes through the dataset.

The display `78/78` refers to batches. Dividing 5,000 images by 64 gives 78
complete batches, with eight images remaining. The code drops that incomplete
batch during training.

## Project 2 utility results

The baseline achieved 25.52 percent training accuracy, 27.4 percent validation
accuracy, and 26.2 percent test accuracy.

DP-SGD achieved 12.5 percent test accuracy, while FADP achieved 9 percent test
accuracy. Because CIFAR-10 has ten classes, random guessing produces about 10
percent accuracy. Therefore, both private models were close to random, and the
FADP model did not learn a reliable classifier in this run.

The training, validation, and test results are all similarly low. This points
to underfitting rather than overfitting. The small network, 5,000-image subset,
and three training epochs were not sufficient for strong CIFAR-10 performance.

The private-model losses were approximately 2.29. Random uniform predictions
on ten classes have a cross-entropy loss of approximately 2.303, which further
confirms that the private models were near random.

## Explanation results

I used Grad-CAM to show which image regions influenced each model's
predictions. I compared private-model heatmaps with baseline heatmaps using
structural similarity, or SSIM, and peak overlap.

DP-SGD obtained an SSIM of 0.108, while FADP obtained 0.273. Peak overlap was
23.2 percent for DP-SGD and 24.3 percent for FADP.

This provides limited support for the feature-preservation idea. FADP's
heatmaps were structurally closer to the baseline heatmaps. However, this must
be interpreted cautiously because the FADP classifier had only 9 percent
accuracy. A stable explanation is not necessarily useful when the prediction
it explains is unreliable.

SSIM and peak overlap measure similarity, not causal correctness. They do not
prove that Grad-CAM reveals the model's true reasoning.

## Privacy results

The calculated DP-SGD epsilon was 1.629. The illustrative FADP epsilon was
6.664. Because smaller epsilon means stronger privacy, DP-SGD has the stronger
reported privacy value.

The FADP value is illustrative rather than a formal privacy guarantee. The
masks are data-dependent and important channels receive less noise, so the
ordinary DP-SGD accountant cannot simply be reused without a more complete
privacy analysis.

I also performed a simple membership-inference attack. Its AUC values were
0.491 for the baseline, 0.473 for DP-SGD, and 0.489 for FADP. All are close to
0.5, which represents chance-level attack performance.

This result does not prove privacy because even the non-private baseline was
near 0.5. The attack may be too weak, or the underfit models may not contain a
strong membership signal.

## Did the experiment confirm the paper's point?

The answer is: only partially.

FADP produced higher Grad-CAM structural similarity than DP-SGD, which gives
limited evidence that adaptive noise preserved some explanation structure.
However, FADP had worse classification accuracy and a larger illustrative
epsilon. It therefore did not provide a better overall privacy–utility–
interpretability balance in this experiment.

This should be reported as a mixed or negative replication result. A
replication is still valuable when it does not confirm the expected result. It
shows that the method's advantage may depend on model architecture, dataset
size, training duration, parameter tuning, and implementation details.

## Limitations and future work

There are several limitations. The experiment used only 5,000 of CIFAR-10's
50,000 training images, trained for only three epochs, used a small CNN rather
than a pretrained MobileNet, and evaluated one random seed. The FADP paper also
does not provide enough implementation detail to reconstruct every decision
exactly.

Future work should use the complete dataset, a stronger architecture, more
epochs, multiple random seeds, and validation-based hyperparameter tuning. It
should also use a stronger membership-inference evaluation and develop a
formal privacy analysis for the adaptive data-dependent masks.

## Conclusion

In conclusion, Project 1 clearly demonstrates that privacy and lossy data
transformation can reduce predictive utility. Project 2 tests whether adaptive
noise can preserve important features. In my run, FADP produced more
structurally similar Grad-CAM explanations, but it did not improve accuracy or
the illustrative privacy bound over DP-SGD.

The main lesson is that adaptive noise is an interesting idea, but it does not
automatically create a better privacy–utility trade-off. It must be evaluated
with reliable models, repeated experiments, and rigorous privacy accounting.

Thank you. I am happy to answer questions.

## Short answers for likely questions

**What is an epoch?**  
One complete pass through the training dataset.

**What is a CNN?**  
An image model that learns visual patterns through convolutional filters.

**What is epsilon?**  
A measure of privacy loss. Smaller epsilon generally means stronger privacy.

**Why did DP-SGD reduce accuracy?**  
Gradient clipping limits learning, and Gaussian noise makes parameter updates
less precise.

**Why was FADP near random accuracy?**  
The private models were underfit under this small, three-epoch configuration.

**Did FADP outperform DP-SGD?**  
It achieved higher explanation SSIM, but lower accuracy and a larger
illustrative epsilon, so it did not outperform DP-SGD overall.

**Does membership AUC near 0.5 prove privacy?**  
No. It only means this particular attack could not distinguish members from
non-members reliably.

**Does the autoencoder provide differential privacy?**  
No. It compresses information but does not provide a formal DP guarantee.

**Did this project prove or disprove the paper?**  
No. It is one small-scale experiment. It produced mixed evidence and identified
conditions that should be tested more rigorously.
