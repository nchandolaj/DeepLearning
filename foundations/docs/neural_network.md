# Neural Networks

*Prerequisites: Basic mathematics and computer science background, with some Python coding experience.*

The best way to think about a **neural network** is as a **universal function approximator**. We are essentially trying to find a massive, complex mathematical function that maps inputs ($x$) to correct outputs ($y$).

## 1. The Neuron Equation: $y = f(\sum(w_i x_i) + b)$

In computer science terms, a neuron is just a function. It takes multiple inputs, performs a weighted sum, adds a constant, and passes the result through a filter.

* **The Input ($\sum w_i x_i$):** This is a dot product between your input vector and your weight vector. It represents the total signal coming into the neuron.
* **The Output ($y$):** This is the final "fire" signal sent to the next layer.

## 2. Weights ($w$): The Strength of Connection

Weights are the "knowledge" of the network. In a program, these are the **parameters** that the algorithm updates during training.

* **Mathematical Role:** They determine the slope or "importance" of an input. If $w_1$ is large and $w_2$ is near zero, the neuron will almost exclusively listen to $x_1$.
* **CS Analogy:** Think of weights as **priority values** in a sorting algorithm or coefficients in a linear regression.

## 3. Bias ($b$): The Activation Threshold

The bias allows you to shift the activation function left or right.

* **Why it matters:** Without a bias, the output of the linear part ($\sum w_i x_i$) would always be $0$ when the input is $0$. This means the "line" would be forced to pass through the origin $(0,0)$.
* **The "Threshold" Concept:** In biology, a neuron needs a certain amount of "pressure" to fire. The bias represents how easy or hard it is to get that neuron to output a positive signal regardless of the inputs.

## 4. Activation ($f$): The Decision Function

If we only had $w$ and $b$, our entire neural network would just be a giant linear equation. Multiple linear layers stacked together still mathematically collapse into a single linear layer. We use **Activation Functions** to introduce **non-linearity**.

* **Sigmoid Function:** $f(z) = \frac{1}{1 + e^{-z}}$. It squashes any input into a range between $0$ and $1$. It’s great for probability.
* **The Purpose:** It decides if the information is important enough to be passed on. It acts like a logic gate that can be "partially open."

# Single Neuron Implementation 

Python code using **NumPy** to implement a single neuron. This script demonstrates the weighted sum, the addition of bias, and the application of the **Sigmoid** activation function.

```python
import numpy as np

def sigmoid(z):
    """The Activation Function (f)"""
    return 1 / (1 + np.exp(-z))

def neuron(inputs, weights, bias):
    """
    The Neuron Equation: y = f(Σ(wi*xi) + b)
    In NumPy, Σ(wi*xi) is efficiently handled by np.dot()
    """
    # 1. Calculate Weighted Sum (z)
    z = np.dot(inputs, weights) + bias
    
    # 2. Apply Activation Function
    y = sigmoid(z)
    
    return y

# --- Example Usage ---

# Inputs (x): e.g., features from a dataset
x = np.array([0.5, 0.8, -0.2])

# Weights (w): The 'learned' strength of each input
w = np.array([0.4, -0.5, 0.1])

# Bias (b): The threshold shift
b = -0.1

output = neuron(x, w, b)

print(f"Inputs: {x}")
print(f"Weights: {w}")
print(f"Bias: {b}")
print(f"---")
print(f"Neuron Output (y): {output:.4f}")
```
