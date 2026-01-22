# Commonly used Activation Functions

In the world of deep learning, **activation functions are the "decision-makers" of a neuron**. 

**Choosing the right one can be the difference between a model that learns in minutes and one that never converges at all.**

Here are the most commonly used functions, the specific hurdles they overcome, and the math behind them.


## 1. ReLU (Rectified Linear Unit)

**The Problem it Solves:** The **Vanishing Gradient Problem** and **Computational Efficiency.**

Before ReLU, functions like Sigmoid were used in hidden layers. However, Sigmoid "flattens out" at high and low values, meaning the gradient (slope) becomes almost zero. This stops the network from learning.

**How it Works:** It is a simple "threshold" function:

$$f(x) = \max(0, x)$$

* If the input is positive, the output is the input.
* If the input is negative, the output is exactly zero.

Because its derivative is always **1** for positive values, the gradient doesn't "vanish" as it travels back through hundreds of layers. It is also incredibly fast to calculate because it only requires a simple comparison.


## 2. Leaky ReLU

**The Problem it Solves:** The **"Dying ReLU" Problem.**

In standard ReLU, if a neuron gets stuck in the negative range, its gradient becomes 0. That neuron effectively "dies" and never updates again.

**How it Works:** It introduces a tiny slope for negative values:

$$f(x) = \max(\alpha x, x)$$ 

(where $\alpha$ is usually a small constant like $0.01$)

By allowing a small, non-zero gradient when $x < 0$, the neuron has a chance to "recover" and start learning again during backpropagation.


## 3. Sigmoid

**The Problem it Solves:** **Binary Classification** and **Probability Mapping.**

When you need **to answer a "Yes or No" question**, you need a value between 0 and 1. 

**How it Works:** It squashes any real-valued number into a range between **0 and 1**:

$$f(x) = \frac{1}{1 + e^{-x}}$$

This makes it perfect for the output layer of a model predicting things like "Is this email spam?" or "Does this image contain a cat?"


## 4. Tanh (Hyperbolic Tangent)

**The Problem it Solves:** **Non-Zero Centered Data.**

Sigmoid outputs are always positive (0 to 1), which can make the weight updates for the next layer always move in the same direction, slowing down training.

**How it Works:** Tanh squashes values between **-1 and 1**:

$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Because it is **zero-centered**, the mean of the activations is closer to zero, which generally **makes the optimization process (Gradient Descent) much more stable and faster than Sigmoid for hidden layers.**


## 5. Softmax

**The Problem it Solves:** **Multi-Class Classification.**

When your model needs to choose between multiple categories (e.g., Is this a Dog, Cat, or Bird?), you need all your outputs to make sense together.

**How it Works:** It takes a vector of raw scores and turns them into a **probability distribution**:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Every output will be between 0 and 1, and **all outputs will sum to exactly 1.0 (100%)**. This allows you to say, "The model is 85% sure this is a Cat."


## 📊 Quick Selection Guide

| Task / Layer | Recommended Function | Why? |
| :--- | :--- | :--- |
| **Hidden Layers** | **ReLU** | Fastest, avoids vanishing gradients. |
| **Hidden Layers (if neurons die)** | **Leaky ReLU** | Prevents dead neurons. |
| **Binary Output** | **Sigmoid** | Maps output to a 0–1 probability. |
| **Multi-Class Output** | **Softmax** | Ensures probabilities sum to 100%. |
| **RNNs / LSTMs** | **Tanh** | Standard for controlling state flow. |

---

# Softmax: Deep Dive

## 🍦 What is Softmax?

In deep learning, **Softmax** is a mathematical function often used as the final activation layer of a neural network. 

Its primary job is to take a vector of raw numerical scores, called **logits**, and transform them into a **probability distribution** where every value is between 0 and 1, and all values sum up exactly to 1.


## 🔢 Converting Logits to Probabilities

When a neural network processes an image (e.g., trying to decide if it's a cat, dog, or bird), the very last linear layer produces "raw scores" for each class. These scores can be any real number—positive, negative, large, or small. We call these **Logits**.

Softmax converts these logits using a specific two-step mathematical process:

1.  **Exponentiation ($e^{z_i}$):** It takes the exponential of each logit. This ensures all values become **positive** and exaggerates the differences between scores (larger scores become much larger).

2.  **Normalization ($\sum e^{z_j}$):** It divides each exponentiated value by the sum of all exponentiated values in the vector.

### The Formula

For a vector of logits $z$, the Softmax value for the $i$-th class is:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$


## 🎯 Example Walkthrough

Imagine a 3-class classification problem (Cat, Dog, Bird). Your network outputs the following **Logits**:
* **Cat:** 2.0
* **Dog:** 1.0
* **Bird:** 0.1

If we simply looked at the raw scores, it's hard to tell "how much" the network prefers the Cat over the Dog. After applying **Softmax**, the output might look like this:
* **Cat:** 0.65 (65% probability)
* **Dog:** 0.24 (24% probability)
* **Bird:** 0.11 (11% probability)
* **Total Sum:** **1.0 (100%)**


## 🚀 Use in the Output Layer

Softmax is almost exclusively used in the **output layer** for multi-class classification tasks. Here is why it is essential:

* **Mutually Exclusive Classes:** It assumes that the input belongs to exactly **one** category. Because the probabilities sum to 1, an increase in the "Cat" probability automatically forces a decrease in the "Dog" and "Bird" probabilities.

* **Interpretability:** It makes the model's output human-readable. Instead of "Score 5.2," we get "92% confidence."

* **Loss Calculation:** Softmax is designed to work perfectly with the **Categorical Cross-Entropy Loss** function. During training, the loss function looks at the probability assigned to the *correct* class; if that probability is low, the backward pass calculates a large gradient to correct the weights.

## ⚠️ Softmax vs. Sigmoid

While both map inputs to a 0–1 range, they serve different purposes:

* **Sigmoid:** Used for **Binary Classification** or **Multi-Label Classification** (where an image can be both a "sunny day" AND "a beach"). Each output is independent.

* **Softmax:** Used for **Multi-Class Classification** (where an image is either a "Cat" OR a "Dog"). All outputs are interdependent.

## Python Code Implementation of Softmax

```python
import numpy as np

def softmax(logits):
    # Step 1: Numerical Stability Trick
    # Subtracting the maximum value prevents the exponents from becoming 
    # infinitely large (overflow), without changing the final result.
    exponents = np.exp(logits - np.max(logits))
    
    # Step 2: Normalization
    # Divide each exponent by the sum of all exponents
    probabilities = exponents / np.sum(exponents)
    
    return probabilities

# Example Raw Scores (Logits) from a CNN
logits = np.array([2.0, 1.0, 0.1])

probs = softmax(logits)

print(f"Logits: {logits}")
print(f"Probabilities: {probs}")
print(f"Sum: {np.sum(probs)}") # Should be exactly 1.0
```
