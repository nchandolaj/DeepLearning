'''
Cross-Entropy Loss, often called Log Loss, is a metric used to measure the performance of a classification model. 
It quantifies the difference between two probability distributions: 
   - the actual labels (the truth) and 
   - the predicted probabilities (the model's guess)

If the probability of the correct class is 1.0, the loss is 0.5 As the predicted probability 
for the correct class decreases toward 0, the loss increases exponentially.

Cross-Entropy Loss is rarely used in isolation; it is almost always paired with the Softmax function.
'''
import numpy as np

def cross_entropy(probabilities, target_index):
    # We only care about the probability of the correct class
    predicted_prob = probabilities[target_index]
    
    # Calculate negative log likelihood
    # Adding a tiny epsilon prevents log(0) which is undefined
    loss = -np.log(predicted_prob + 1e-9)
    
    return loss

# Let's say the correct label was 'Cat' (index 0)
target_idx = 0 
probs = [0.65900114, 0.24243297, 0.09856589]
loss_value = cross_entropy(probs, target_idx)

print(f"Probabilities: {probs}")
print(f"Correct Class Probability: {probs[target_idx]:.4f}")
print(f"Cross-Entropy Loss: {loss_value:.4f}")