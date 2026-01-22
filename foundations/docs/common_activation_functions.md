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


