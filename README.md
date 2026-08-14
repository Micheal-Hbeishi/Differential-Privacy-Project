# Recreating Privacy, Utility, and Explainability in Image Classification

This is my simplified recreation of the main idea from Abbasi, Mori, and
Saracino's paper, *Trading-Off Privacy, Utility, and Explainability in Deep
Learning-Based Image Data Analysis*.

The paper studies a trade-off: increasing privacy can make a model less
accurate, while explainability helps us see why the model made a decision. My
project demonstrates those ideas on MNIST handwritten-digit images.

## What I implemented

- A normal CNN classifier as the accuracy baseline.
- A TensorFlow Privacy DP-SGD classifier that clips gradients and adds noise.
- An autoencoder that compresses and reconstructs input images.
- A SmoothGrad saliency map that highlights pixels used for a prediction.
- A CSV comparison of accuracy, privacy information, and utility loss.

## Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python notebook/simple_paper_replication.py
```

MNIST is downloaded automatically the first time the program runs. The script
uses 5,000 training images and 1,000 test images so it can run on a regular
laptop. These are deliberate simplifications, not the full scale of the paper.

## Outputs

The script creates these files in `results/simple_paper_replication/`:

- `comparison.csv` - accuracy and privacy/utility comparison.
- `smoothgrad_example.png` - one digit and the corresponding explanation map.
- `run_metadata.json` - the run settings used to create the results.

## How it connects to the paper

| Paper idea | My simplified version |
|---|---|
| Image classification | MNIST digit classifier |
| Differential privacy | TensorFlow Privacy DP-SGD |
| Data anonymization | Autoencoder bottleneck and reconstruction |
| Explainability | SmoothGrad saliency map |
| Privacy/utility comparison | Accuracy and privacy information in `comparison.csv` |

This project recreates the core workflow, not every experiment in the paper.
The paper also evaluates FER and CIFAR-10, uses larger experiments, and
considers a multi-stakeholder compatibility matrix.

## Project files

```text
notebook/simple_paper_replication.py  # Main project to run
results/simple_paper_replication/     # Files produced by the main project
PROJECT_NOTES.md                      # Short explanation for the project/presentation
```
