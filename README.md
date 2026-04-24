# 🧠 Learnable Weight Pruning using PyTorch

This project implements a **neural network with learnable pruning** using PyTorch, where the model automatically decides which weights to keep or remove during training.

Instead of traditional pruning (post-training), this approach integrates pruning **directly into the learning process** using trainable gates.

---

## 🚀 Overview

* Each weight is paired with a **learnable gate parameter**

* Gates are passed through a **sigmoid function** → values in (0,1)

* Effective weight:

  ```
  pruned_weight = weight × sigmoid(gate_score)
  ```

* A **sparsity loss (L1 penalty)** is applied to push gates toward 0

---

## 🏗️ Model Architecture

Fully connected neural network for CIFAR-10:

```
Input (3072)
   ↓
512 → 256 → 128 → 10
```

Each layer includes:

* Custom `PrunableLinear`
* Batch Normalization
* ReLU Activation
* Dropout (for regularization)

---

## ⚙️ Key Components

### 1. PrunableLinear Layer

* Drop-in replacement for `nn.Linear`
* Adds learnable gates to each weight
* Enables **differentiable pruning**

---

### 2. Sparsity Loss

```
Total Loss = CrossEntropy + λ × L1(gates)
```

* λ controls pruning strength:

  * Small λ → higher accuracy
  * Large λ → higher sparsity

---

### 3. Training Strategy

* Optimizer: Adam
* Learning Rate Scheduler: Cosine Annealing
* Dataset: CIFAR-10
* Data Augmentation:

  * Random crop
  * Horizontal flip

---

## 📊 Evaluation Metrics

* **Accuracy (%)** → Classification performance
* **Sparsity (%)** → % of weights pruned

A weight is considered pruned if:

```
gate < 0.01
```

---

## 🔁 Experiments

The model is trained with multiple λ values:

```python
LAMBDAS = [1e-5, 1e-4, 1e-3]
```

For each λ:

1. Train model
2. Evaluate accuracy
3. Compute sparsity
4. Plot gate distribution

---

## 📈 Output

### ✔ Console Results

* Training loss
* Accuracy
* Sparsity

### ✔ Visualization

The script generates:

```
gate_distribution.png
```

This shows:

* Gate value distribution
* Effect of λ on pruning
* Threshold line for sparsity

---

## 📁 Project Structure

```
.
├── main.py
├── data/
├── gate_distribution.png
└── README.md
```

---

## ⚡ Installation

```bash
pip install torch torchvision matplotlib numpy
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🧪 Key Insights

* Increasing λ increases sparsity
* Too much pruning reduces accuracy
* Model learns which connections are important

---

## 🔧 Customization

You can modify:

* `LAMBDAS` → pruning strength
* `NUM_EPOCHS` → training duration
* `BATCH_SIZE` → performance
* `gate_threshold` → sparsity definition

---

## 💡 Future Work

* Apply to CNN architectures
* Structured pruning (filter/channel level)
* Model compression for deployment
* Compare with magnitude pruning

---

## 📌 Conclusion

This project demonstrates how pruning can be made **learnable and differentiable**, allowing the model to automatically optimize both performance and efficiency.

---

## 👩‍💻 Author

Hamsha Vardhini

---

⭐ If you found this useful, consider starring the repo!
