# Perceptron

A **Perceptron** is the fundamental building block of modern Artificial Intelligence. Invented by Frank Rosenblatt in 1958, it is the simplest form of a neural network used for binary classification—deciding whether an input belongs to one class or another.

Think of it as a mathematical model of a biological neuron: it takes several inputs, processes them, and produces a single output.

## 1. Components of a Perceptron

To understand how a Perceptron "thinks," we need to look at its four main internal components.

### A. Inputs ($x_1, x_2, ..., x_n$)
These are the features of your data. If you are trying to predict if a fruit is an orange, the inputs might be weight, texture, and color.

### B. Weights ($w_1, w_2, ..., w_n$)
Weights are the most critical part of the network. They represent the **strength** or importance of each input. 
* A high weight means that specific input has a big impact on the output.
* During training, the Perceptron adjusts these weights to reduce errors.

### C. Bias ($b$)
The bias is an extra parameter that allows the model to shift the activation function up or down. It ensures that even if all inputs are zero, the neuron can still produce an output. Mathematically, it acts like the intercept in a linear equation ($y = mx + b$).

### D. Summation and Activation Function
1.  **Summation:** The Perceptron calculates the "Weighted Sum" of the inputs:
    $$Z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$
2.  **Activation:** This sum is passed through an **Activation Function** (usually a Step Function). If the result is above a certain threshold, the Perceptron "fires" (outputs 1); otherwise, it outputs 0.

## 2. Key Characteristics
* **Linear Classifier:** A single Perceptron can only classify data that is **linearly separable** (data that can be split by a single straight line).
* **Binary Output:** It typically results in a 1 or 0 (Yes/No).
* **Supervised Learning:** It learns by comparing its guess to the actual answer and adjusting weights accordingly.

## 3. Significance in Deep Learning
While a single Perceptron is limited (it famously failed the "XOR problem" because it couldn't handle non-linear data), it remains vital because:
* **The Foundation:** Modern Deep Learning involves stacking thousands of these units into "Multi-Layer Perceptrons" (MLPs).
* **Proof of Concept:** It proved that machines could learn from data iteratively.
* **Basis for Neural Networks:** Every complex architecture today, from ChatGPT to computer vision models, is essentially a massive evolution of the Perceptron.

## 4. Python Implementation (NumPy)

Here is a simple implementation using `NumPy` to simulate a logical **AND Gate**.

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, x):
        # Heaviside Step Function
        return 1 if x >= 0 else 0

    def train(self, X, y):
        # Initialize weights to zeros based on number of features
        self.weights = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                # Calculate Weighted Sum (Linear Output)
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)
                
                # Perceptron Learning Rule: Update weights based on error
                error = y[i] - prediction
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(i) for i in linear_output])

# Data for AND Gate (Linearly Separable)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Training the model
model = Perceptron()
model.train(X, y)

# Testing
predictions = model.predict(X)
print(f"Final Weights: {model.weights}")
print(f"Final Bias: {model.bias}")
print(f"Predictions: {predictions}")
```
