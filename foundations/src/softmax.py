'''
In deep learning, Softmax is a mathematical function often used as the final activation layer of a neural network. 
Its primary job is to take a vector of raw numerical scores, called logits, and transform them into a probability distribution 
where every value is between 0 and 1, and all values sum up exactly to 1.
'''

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