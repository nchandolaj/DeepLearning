'''
Goal: Understand the Matrix Operations for Deep learning. Demonstrate understanding of vectorizations, shapes, and equations.
- Task 1: Initialize a "dummy" network with random weights using np.random.randn.
- Task 2: Create a random input batch (e.g., 3 samples, 4 features).
- Task 3: Perform the forward pass calculation Z = X ⋅ W + B and print the shape of the result. Verify it matches your expectation.

Use Case: Credit Scoring using a dummy network. We have 3 applicants (our Batch Size), 
and for each applicant, we are looking at 4 features:
- Credit Score
- Annual Income
- Debt-to-Income Ratio
- Employment Duration

We want to pass this through a "Hidden Layer" of 2 neurons that will attempt to find patterns in the applicants' data.
'''

import numpy as np

# --- Task 1: Initialize a "dummy" network with random weights ---
# Architecture: 4 input features -> 2 hidden neurons
input_features = 4
hidden_neurons = 2

# We use np.random.randn to get values from a standard normal distribution
W = np.random.randn(input_features, hidden_neurons)
B = np.random.randn(hidden_neurons)

# --- Task 2: Create a random input batch ---
# Scenario: 3 applicants (samples), 4 features each
batch_size = 3
X = np.random.randn(batch_size, input_features)

print("--- Data Shapes ---")
print(f"Input Batch X: {X.shape} (Samples: 3, Features: 4)")
print(f"Weight Matrix W: {W.shape} (Input: 4, Neurons: 2)")
print(f"Bias Vector B:   {B.shape} (One per Neuron: 2)")

# --- Task 3: Perform the Forward Pass (Z = XW + B) ---
# We use '@' for matrix multiplication (dot product)
Z = (X @ W) + B

print("\n--- Result ---")
print(f"Result Z:\n{Z}")
print(f"Shape of Z: {Z.shape}")

# --- Verification ---
expected_shape = (batch_size, hidden_neurons)
if Z.shape == expected_shape:
    print(f"\nVerification Success: Result matches {expected_shape}")
else:
    print("\nVerification Failed: Dimensions do not match.")