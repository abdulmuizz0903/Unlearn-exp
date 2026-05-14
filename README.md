# MNIST Targeted Machine Unlearning

This project implements a mechanistic approach to Machine Unlearning on simple Multilayer Perceptron (MLP) architectures trained on the MNIST dataset. The goal is to "unlearn" a specific class (in this case, the digit `3`) without retraining the model from scratch, while preserving the accuracy on the rest of the classes.

## Methodology: Activation Penalization
The unlearning algorithm isolates the specific neurons responsible for representing the targeted class:
1. **Activation Profiling**: We pass a "Forget Set" (images of the target digit) through the network and compute the mean activation of the neurons in the final hidden layer.
2. **Targeted Pruning**: We isolate the most active neurons (e.g., top 10% active neurons for the target digit).
3. **Weight Penalization**: We heavily penalize (scale down) the incoming and outgoing weights of these highly active neurons. Over iterative passes, the model cleanly dissects and destroys the feature representation of the targeted class.

---

## Dataset Splits
This experiment uses the standard MNIST dataset (60,000 training images, 10,000 test images). 
For the purposes of unlearning, we partition the dataset dynamically:

* **Target Digit**: `3`
* **Overall Test Set**: All 10,000 MNIST test images.
* **Forget Set**: Only the images of the target digit (`3`). 
  * *Size*: 1,010 images.
* **Retain Set**: All images of digits except the target digit (`0-2, 4-9`). 
  * *Size*: 8,990 images.

> *Note: If a model perfectly unlearns the target digit (0% accuracy on the Forget Set) while retaining 100% accuracy on the Retain Set, the mathematical ceiling for Overall Test Accuracy is exactly **89.90%**.*

---

## Findings

We executed unlearning across single-layer and multi-layer MLP architectures to observe how structural capacity and depth affect feature disentanglement.

### single-Layer MLP Architecture
For single-layer MLPs, unlearning is fundamentally challenging because neurons are highly polysemantic (a single neuron encodes parts of different digits). 

| Hidden Size | Base Overall | Base Retain | Base Forget | Post Overall | Post Retain | Post Forget |
|---|---|---|---|---|---|---|
| 64 | 96.56% | 96.42% | 97.82% | 62.86% | 69.72% | 1.78% |
| 128 | 97.29% | 97.31% | 97.13% | 67.15% | 74.33% | 3.27% |
| 256 | 97.51% | 97.62% | 96.53% | 78.05% | 86.42% | 3.56% |
| 512 | 97.63% | 97.62% | 97.72% | 78.84% | 86.71% | 8.81% |
| 1024 | 97.63% | 97.71% | 96.93% | 80.80% | 89.19% | 6.14% |

* **Observation**: Smaller architectures fail to isolate the digit, leading to massive collateral damage (Retain accuracy drops to ~69%). Wider networks (1024) have spare capacity, allowing for cleaner, class-specific node targeting (Retain accuracy ~89%), but still struggle to perfectly eliminate knowledge of `3`.

### Multi-Layer (Deep) Architecture
We applied the exact same unlearning constraints (targeting the *last* hidden representation layer before the classification head) across 2-layer and 3-layer networks. 

| Hidden Layers | Base Overall | Base Retain | Base Forget | Post Overall | Post Retain | Post Forget |
|---|---|---|---|---|---|---|
| [256, 128] | 97.30% | 97.22% | 98.02% | 82.69% | 89.72% | 20.10% |
| [512, 256] | 97.75% | 97.73% | 97.92% | 84.68% | 94.10% | 0.79% |
| [1024, 512] | 97.61% | 97.62% | 97.52% | 87.60% | 97.01% | 3.86% |
| [256, 128, 64] | 97.37% | 97.43% | 96.83% | 77.13% | 84.20% | 14.16% |
| [512, 256, 128] | 97.29% | 97.10% | 99.01% | 85.20% | 94.03% | 6.63% |
| **[1024, 512, 256]** | 97.51% | 97.39% | 98.61% | **87.25%** | **97.05%** | **0.00%** |

* **Observation**: Depth drastically promotes feature disentanglement. The `[1024, 512, 256]` sequence was incredibly successful. After unlearning operations, we achieved exactly **0.00% accuracy on the target digit**, yet largely preserved the retain performance at **97.05%**. This approaches the theoretical maximum (89.9%) for a flawless surgical unlearning deletion.

## Running the Code
1. Ensure the `Unlearn` Conda environment is active (Requires PyTorch and Torchvision).
2. Run single-layer evaluation: `python run_evaluation.py`
3. Run wide/deep evaluation: `python run_deep_eval.py`