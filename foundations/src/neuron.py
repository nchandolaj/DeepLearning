'''
A Single Neuron, or Perceptron with an activation function.
Reference: https://github.com/nchandolaj/DeepLearning/blob/main/foundations/docs/neural_network.md
'''
import numpy as np
from sigmoid import sigmoid

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

if __name__ == "__main__":
    # Inputs (x): e.g., features from a dataset
    x = np.array([0.5, 0.8, -0.2])

    # Weights (w): The 'learned' strength of each input
    w = np.array([0.4, -0.5, 0.1])

    # Bias (b): The threshold shift
    b = -0.1

    print(neuron(x, w, b))