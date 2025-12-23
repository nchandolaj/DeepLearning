# Multi-Layer Perception (MLP)

**The "XOR Problem":** In the 1960s, it was discovered that a single-layer neuron (a Perceptron) could only solve problems that are **linearly separable**.

### The Logic Gap
* **OR/AND:** You can draw a single straight line on a graph to separate $(0,0)$ from $(1,1)$.
* **XOR (Exclusive OR):** The outputs for $(0,1)$ and $(1,0)$ are true, while $(0,0)$ and $(1,1)$ are false. You **cannot** draw one straight line to separate these two groups.

**The XOR Problem in Deep Learning:** The XOR (Exclusive OR) problem is a classic milestone in AI history. It represents the moment we realized that "Deep" (multi-layer) networks were necessary to solve non-linear problems.

## 1. Why a Single Neuron Fails (The Geometry)

In basic logic gates like **AND** or **OR**, you can visualize the inputs $(x_1, x_2)$ on a 2D plane. 

* **AND Gate:** Only $(1,1)$ is True. You can easily draw a straight line that keeps $(1,1)$ on one side and $(0,0), (0,1), (1,0)$ on the other.
* **XOR Gate:** $(0,1)$ and $(1,0)$ are True (Output 1), while $(0,0)$ and $(1,1)$ are False (Output 0).

If you plot these points, the "True" results are diagonal to each other. There is **no way** to draw a single straight line that separates the 1s from the 0s. 

Mathematically, a single neuron equation $y = f(w_1x_1 + w_2x_2 + b)$ represents a **linear decision boundary** (a straight line). Since a single line cannot separate these points, a single neuron's accuracy will never exceed 75% for the XOR problem.

## 2. The Solution: Multi-Layer Perceptron (MLP)

To solve XOR, we need to move from 2D space into a higher-dimensional space where the data *becomes* linearly separable. We do this by adding a **Hidden Layer**.

### The Logic of the Solution
Think of XOR as a combination of other gates. 
$XOR(A, B) = (A \text{ OR } B) \text{ AND NOT } (A \text{ AND } B)$

In a neural network, we use a hidden layer to perform this transformation:
1.  **Hidden Layer:** It takes the 2 inputs and warps the coordinate system.
2.  **Output Layer:** In this new warped space, the points are now "linearly separable," meaning a single straight line can finally divide them.



---

## 3. Python Implementation (NumPy Solution)

Below is a manual implementation of a 2-layer network. These weights are pre-calculated to show how the logic works.

```python
import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# XOR Inputs
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Manual Weights for a solution
# Hidden Layer (2 neurons)
# These weights transform the input into a new space
w_hidden = np.array([[20, -20], [20, -20]]) 
b_hidden = np.array([-10, 30])

# Output Layer (1 neuron)
# This neuron performs the final linear separation
w_output = np.array([[20], [20]])
b_output = -30

def forward_pass(inputs):
    # Layer 1: Hidden
    z1 = np.dot(inputs, w_hidden) + b_hidden
    a1 = sigmoid(z1)
    
    # Layer 2: Output
    z2 = np.dot(a1, w_output) + b_output
    a2 = sigmoid(z2)
    return a2

# Test the solution
print("XOR Prediction Results:")
for i in range(len(X)):
    pred = forward_pass(X[i])
    print(f"Input: {X[i]} -> Predict: {pred[0]:.4f}")
