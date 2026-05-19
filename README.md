# MNIST Targeted Machine Unlearning

This project implements a mechanistic approach to Machine Unlearning on simple Multilayer Perceptron (MLP) architectures trained on the MNIST dataset. The goal is to "unlearn" a specific class (in this case, the digit `3`) without retraining the model from scratch, while preserving the accuracy on the rest of the classes.

## Methodology: Activation Penalization

The unlearning algorithm employs a Mechanistic Iterative Node Ablation technique. Given a pre-trained neural network $f_\theta$, we isolate and penalize the specific parameter pathways responsible for representing the targeted class knowledge.

### Formal Mathematical Formulation

Let $\mathcal{D}_f = \{x_i, y_i\}_{i=1}^N$ represent the **Forget Set** containing strictly the target class $c_f$ (e.g., the digit $3$). Let the target hidden layer map inputs to an activation vector $a(x) \in \mathbb{R}^H$, parameterized by incoming weights $W_{in}$ and bias $b_{in}$, and mapping to subsequent layers via outgoing weights $W_{out}$.

**1. Activation Profiling:**  
We compute the expected activation $\mu_j$ for each neuron $j \in \{1, \dots, H\}$ over the forget set $\mathcal{D}_f$:

$$
\mu_j = \mathbb{E}_{x \sim \mathcal{D}_f}[a_j(x)] = \frac{1}{N} \sum_{i=1}^{N} a_j(x_i)
$$

**2. Targeted Masking (Neuron Selection):**  
We define a threshold $\tau$ corresponding to the $q$-th percentile (e.g., $q=90$) of the empirical distribution of $\mu$. We then construct a binary mask $m \in \{0, 1\}^H$, mathematically isolating the top highly active neurons responsible for expressing the forget class:

$$
m_j = \begin{cases} 1, & \text{if } \mu_j \geq \tau \\ 0, & \text{otherwise} \end{cases}
$$

Let $\mathcal{V}_{active} = \{j \mid m_j = 1\}$ denote the targeted subset of neurons.

**3. Iterative Weight Penalization:**  
To systematically suppress the network's predictive capabilities for $c_f$, we directly scale the parameters contributing to the forward-pass and readout of $\mathcal{V}_{active}$. Given a penalty scaling factor $\lambda \in [0, 1)$ (e.g., $\lambda = 0.1$), for each iteration block $t = 1 \dots T$, we apply the following multiplicative decay to the specific weights associated with the targeted neurons $j \in \mathcal{V}_{active}$:

$$
\begin{aligned}
W_{in}^{(t)}[j, :] &\leftarrow \lambda \cdot W_{in}^{(t-1)}[j, :] \\
b_{in}^{(t)}[j] &\leftarrow \lambda \cdot b_{in}^{(t-1)}[j] \\
W_{out}^{(t)}[:, j] &\leftarrow \lambda \cdot W_{out}^{(t-1)}[:, j]
\end{aligned}
$$

Over $T$ iterative passes (which forces continuous re-profiling of fallback features), the model geometrically collapses the embedded manifold of the targeted class while generally leaving disjoint feature spaces (the Retain Set) intact.

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