# Parameters and Hyperparameters

In the context of Deep Networks, the distinction between **Parameters** and **Hyperparameters** is crucial. Think of the network as a machine you are building and training.

* **Parameters** are the internal parts of the machine that move and adjust themselves automatically.
* **Hyperparameters** are the settings on the control panel that *you* (the human) must set before you turn the machine on.


## 1. What are Parameters? (The "Learned Knowledge")
Parameters are the internal variables that the model learns from the training data. They are the values that change during the "training" process via backpropagation.

* **Definition:** Variables optimized by the algorithm to minimize the loss function.
* **Who sets them?** The Model (automatically).
* **When are they set?** During the training phase (they start random and get refined).
* **Examples:**
    * **Weights ($W$):** The strength of the connection between two neurons.
    * **Biases ($b$):** The threshold or offset for a neuron.

> **Analogy:** If a neural network is a student studying for an exam, the **Parameters** are the connections in the student's brain (synapses) that strengthen as they learn the material.


## 2. What are Hyperparameters? (The "Configuration")
Hyperparameters are the external configurations that you define *before* training begins. They control *how* the model learns and what the model's structure looks like. The model cannot change these on its own.

* **Definition:** Settings that govern the training process and the architecture of the model.
* **Who sets them?** You (the Data Scientist/Engineer).
* **When are they set?** Before training starts.
* **Examples:**
    * **Learning Rate:** How big of a step the model takes during optimization (e.g., 0.001 vs 0.01).
    * **Number of Epochs:** How many times the model sees the entire dataset.
    * **Batch Size:** How many data examples the model processes at once.
    * **Architecture:** Number of layers, number of neurons per layer, type of activation function.

> **Analogy:** In the student analogy, the **Hyperparameters** are the study environment: "How many hours will they study?" (Epochs), "Will they study alone or in a group?" (Batch Size), "How fast will they try to read?" (Learning Rate).


## 3. Comparison Table

| Feature | Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Origin** | Learned from data | Set manually by human (or tuning algos) |
| **Status** | Internal to the model | External to the model |
| **Final State** | Saved as part of the trained model file | Not saved in the model file (only used to create it) |
| **Example** | Weights ($W$), Biases ($b$) | Learning Rate ($\alpha$), # of Layers, Dropout rate |
| **How to optimize** | Backpropagation / Gradient Descent | Grid Search, Random Search, Bayesian Optimization |


## 4. Why the distinction matters
If you have a "bad" model, you need to know which knob to turn:
1.  **Underfitting (Model is too dumb):** You might need to change the **Hyperparameters** (add more layers, train longer).
2.  **Overfitting (Model memorizes data):** You might need to change **Hyperparameters** (add Dropout, increase regularization) or get more data to help the **Parameters** generalize better.

The process of finding the best hyperparameters is called **"Hyperparameter Tuning."** It is often trial-and-error, whereas finding the best parameters is purely mathematical (Calculus).

---

# "Must-Know" Hyperparameters

When building deep networks, you can easily get overwhelmed by the sheer number of settings. However, you usually only need to worry about a few critical ones to get a model working.

Here are the **"Must-Know" Hyperparameters**, ranked by how likely they are to make or break your model.


## 1. The Optimization Trio (The Engine Settings)

These control *how* the model learns. If these are wrong, the model will either never learn or take forever.

### **A. Learning Rate ($\alpha$ or `lr`)**
* **What it is:** The size of the step the optimizer takes when updating the weights.
* **Why it matters:** It is widely considered the **single most important hyperparameter**.
* **The Goldilocks Zone:**
    * **Too Low:** The model learns incredibly slowly (it might take years to converge).
    * **Too High:** The model overshoots the target, bounces around, and eventually diverges (loss goes to infinity).
    * **Typical Values:** `0.1`, `0.01`, `0.001` (start here), `0.0001`.


### **B. Batch Size**
* **What it is:** The number of training examples the model sees before it updates its parameters once.
* **Why it matters:** It balances speed vs. stability.
    * **Small Batch (e.g., 8, 16):** Noisy updates (the path to the minimum is jagged), but often generalizes better. Uses less memory.
    * **Large Batch (e.g., 256, 1024):** Stable, smooth updates. Faster training on GPUs, but requires massive VRAM and can sometimes get stuck in "sharp minima" (bad generalization).
    * **Typical Values:** Powers of 2 (`32`, `64`, `128`) are preferred for computer memory alignment.

### **C. Number of Epochs**
* **What it is:** The number of times the model sees the *entire* dataset.
* **Why it matters:** Controls how long you train.
    * **Too Few:** Underfitting (the model hasn't seen enough data to learn patterns).
    * **Too Many:** Overfitting (the model starts memorizing the specific examples instead of general rules).
* **Strategy:** Set this high (e.g., 100) but use **"Early Stopping"**—a mechanism that automatically kills the training when the model stops improving.


## 2. The Architecture settings (The Blueprint)

These control the *shape* and *capacity* of the model.

### **D. Number of Hidden Layers (Depth)**
* **What it is:** How many layers exist between input and output.
* **Rule of Thumb:** Start small (1–2 layers). Go deeper only if the problem is complex (like image recognition).
* **Effect:** deeper networks can learn more complex features but are harder to train (vanishing gradients).

### **E. Number of Neurons per Layer (Width)**
* **What it is:** The size of the hidden layers.
* **Rule of Thumb:** Common sizes are `64`, `128`, `256`, or `512`.
* **Effect:** Wider networks can capture more details but are more prone to overfitting and take longer to compute.


## 3. The Regularization Settings (The Guardrails)

These prevent the model from cheating (overfitting).

### **F. Dropout Rate**
* **What it is:** The probability (percentage) of randomly "turning off" a neuron during a training step.
* **Why it matters:** It forces the network to be redundant and robust, preventing it from relying too much on any single neuron.
* **Typical Values:** `0.2` (20%) to `0.5` (50%).

### **G. Weight Decay (L2 Regularization)**
* **What it is:** A penalty added to the loss function for having large weights.
* **Why it matters:** It keeps the weights small, which generally leads to simpler, smoother models that generalize better to new data.
* **Typical Values:** `1e-4`, `1e-5`.


## Summary Cheat Sheet

| Hyperparameter | Start Value (Rule of Thumb) | If Model Underfits (High Bias) | If Model Overfits (High Variance) |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | `0.001` (Adam Optimizer) | Increase | Decrease |
| **Batch Size** | `32` or `64` | Decrease (adds noise) | Increase (sometimes helps) |
| **Layers (Depth)** | 2–3 layers | **Increase** | Decrease |
| **Neurons (Width)**| `128` | **Increase** | Decrease |
| **Dropout** | `0.0` (None) | Decrease (remove it) | **Increase** (add it) |
| **Epochs** | 20–50 | Increase | Decrease (or Early Stop) |


