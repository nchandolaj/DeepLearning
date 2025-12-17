# Week 1: The Foundations - Neural Networks from Scratch

**Goal:** Build a fully functional Multi-Layer Perceptron (MLP) using only NumPy to classify digits.
**Prerequisite Check:** Ensure you have Python installed with `numpy`, `matplotlib`, and `sklearn` (just for loading data).

---

## Day 1: The Perceptron (The Atomic Unit)
**Focus:** Understanding the mathematical model of a single neuron.

* **Concept:** The Neuron equation: $y = f(\sum(w_i x_i) + b)$.
    * **Weights ($w$):** The strength of the connection.
    * **Bias ($b$):** The activation threshold.
    * **Activation ($f$):** The decision function.
* **The History:** Read about the "XOR Problem" – why a single layer cannot solve non-linear problems.
* **Resource:** [Neural Networks and Deep Learning (Ch 1)](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielsen.
* **Task:** Write a raw Python function (no NumPy yet) that takes lists `x`, `w`, and float `b` and calculates the output.

## Day 2: The Forward Pass (Matrix Magic)
**Focus:** scaling from one neuron to thousands using Linear Algebra.

* **Concept:** Replacing `for` loops with Dot Products.
    * Layer Equation: $Z = X \cdot W + B$
    * Activation: $A = \sigma(Z)$
* **Shapes matter:** If Input $X$ is `(1, 784)` and Hidden Layer has 128 neurons, what shape is $W$? (Answer: `784x128`).
* **Task:** Initialize a "dummy" network with random weights and pass one image through it.
    ```python
    import numpy as np
    # Random input (1 sample, 4 features)
    x = np.random.randn(1, 4)
    # Weights connecting 4 inputs to 3 hidden neurons
    W1 = np.random.randn(4, 3)
    # Dot product
    z = np.dot(x, W1)
    print(z.shape) # Should be (1, 3)
    ```

## Day 3: Activation & Loss Functions
**Focus:** Non-linearity and measuring error.

* **Activations:**
    * **ReLU:** $max(0, z)$ (The standard for hidden layers).
    * **Softmax:** Converts raw scores (logits) into probabilities (sum = 1).
* **Loss Functions:**
    * **MSE (Mean Squared Error):** For regression (predicting house prices).
    * **Cross-Entropy:** For classification (predicting "Cat" vs "Dog").
* **Task:** Implement `softmax(x)` and `cross_entropy_loss(y_pred, y_true)` in NumPy. Test them with dummy values.

## Day 4: The Chain Rule (Backpropagation Theory)
**Focus:** The most difficult concept in Deep Learning. Do not skip.

* **The "Why":** We need to know how changing a specific weight $w$ affects the final Loss $L$. This is the gradient: $\frac{\partial L}{\partial w}$.
* **The Calculus:**
    1.  How Loss changes w.r.t Output.
    2.  How Output changes w.r.t Activation.
    3.  How Activation changes w.r.t Weighted Sum.
    4.  Multiply them all together (Chain Rule).
* **Video:** Watch Andrej Karpathy's **"The spelled-out intro to neural networks and backpropagation (Micrograd)"** (First 45 mins).
* **Visual:** Draw the computation graph for $f(x,y,z) = (x+y)*z$ and calculate gradients by hand.

## Day 5: Optimizer (Stochastic Gradient Descent)
**Focus:** Updating the weights to learn.

* **Concept:** $W_{new} = W_{old} - (LearningRate \times Gradient)$
* **Learning Rate ($\alpha$):** Too high = overshoot; Too low = never converge.
* **Batches:**
    * **Batch GD:** Use all data for one update (slow, stable).
    * **SGD:** Use one sample for one update (fast, noisy).
    * **Mini-batch:** The best of both worlds (e.g., 32 images at a time).

## Day 6: Project Part I - Building the Engine
**Focus:** Assembling the class structure.

* **Task:** Create `neural_net.py`.
* **Code Structure:**
    ```python
    class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            # Init weights (He Initialization is best, but random is fine for now)
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))

        def forward(self, X):
            # Z1 = X dot W1 + b1
            # A1 = ReLU(Z1)
            # Z2 = A1 dot W2 + b2
            # A2 = Softmax(Z2)
            # Store values for backward pass!
            return result
    ```

## Day 7: Project Part II - Training on MNIST
**Focus:** The "Hello World" of Deep Learning.

* **Dataset:** Use `sklearn.datasets.load_digits()` (easy version) or download full MNIST.
* **The Training Loop:**
    1.  Shuffle data.
    2.  Split into mini-batches.
    3.  **Forward Pass**: Calculate predictions.
    4.  **Compute Loss**: Print it every 10 epochs.
    5.  **Backward Pass**: Calculate gradients (implement the derivatives derived on Day 4).
    6.  **Update Weights**: Apply SGD.
* **Success Metric:** You should see the Loss decrease (e.g., 2.3 -> 1.5 -> 0.4) and Accuracy increase to >85%.
* **Reflection:** Notice how "fiddly" the learning rate is. If you set it to `0.1`, it might work. If `10.0`, it explodes.
