# Project Notes

## Goal

I recreated the paper's main privacy, utility, and explainability ideas in a
smaller project that I can run locally and explain clearly.

## My workflow

1. Train a normal CNN on MNIST and record its accuracy.
2. Train the same type of CNN with TensorFlow Privacy DP-SGD.
3. Compare the private model's accuracy with the normal model's accuracy.
4. Train an autoencoder to compress and reconstruct the MNIST images.
5. Use SmoothGrad to show which pixels influenced a classification.

## What to explain in a presentation

- DP-SGD protects training examples by clipping each example's gradient and
  adding Gaussian noise before the model update.
- Epsilon is the privacy budget: lower epsilon means stronger privacy.
- The autoencoder hides some image detail by passing the image through a
  smaller 32-value representation.
- SmoothGrad averages gradients from slightly noisy copies of the same image,
  producing a cleaner explanation map.
- Utility loss is the reduction in accuracy or image quality caused by the
  privacy technique.

## Scope

This is a simplified reproduction of the workflow. It uses a smaller MNIST
subset and one privacy setting so that the code remains understandable and can
run on a typical laptop. It should not be described as an exact numerical
reproduction of every result in the original paper.
