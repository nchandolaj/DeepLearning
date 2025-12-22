# Week 1: The Foundations - Neural Networks from Scratch

**Goal:** Build a fully functional Multi-Layer Perceptron (MLP) using only NumPy (no PyTorch/TensorFlow) to classify digits.
**Prerequisite Check:** Ensure you have Python installed with `numpy`, `matplotlib`, and `sklearn` (for data loading only).

---

## Day 1: The Perceptron (The Atomic Unit)
**Focus:** Understanding the mathematical model of a single neuron and why we need non-linearity.

* **1. Concepts**
    * The Neuron Equation: $y = f(\sum(w_i x_i) + b)$.
    * **Weights ($w$):** The strength of the connection.
    * **Bias ($b$):** The activation threshold (shifting the line).
    * **Activation ($f$):** The decision function (e.g., Sigmoid).
    * The "XOR Problem": Why a single layer cannot solve non-linear problems.

* **2. Coding Practice**
    * **Task 1:** Write a raw Python function `sigmoid(x)` that returns $1 / (1 + e^{-x})$.
    * **Task 2:** Write a function `neuron(inputs, weights, bias)` that computes the dot product and applies the sigmoid.
    * **Task 3:** Manually find weights/bias that make a single neuron function like an "AND" gate (inputs 0/1).

* **3. Reading**
    * [Neural Networks and Deep Learning (Ch 1)](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielsen (Read up to "Sigmoid Neurons").

* **4. Resource(s)**
    * 

[Image of biological vs artificial neuron]

    * Documentation: [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

---

## Day 2: The Forward Pass (Matrix Magic)
**Focus:** Scaling from one neuron to thousands using Linear Algebra to avoid slow loops.

* **1. Concepts**
    * **Vectorization:** Replacing `for` loops with Matrix Multiplication.
    * **Shapes:** Understanding dimensions. If Input $X$ is `(Batch_Size, Input_Features)` and Weights $W$ is `(Input_Features, Hidden_Neurons)`, the result is `(Batch_Size, Hidden_Neurons)`.
    * The Layer Equation: $Z = X \cdot W + B$.

* **2. Coding Practice**
    * **Task 1:** Initialize a "dummy" network with random weights using `np.random.randn`.
    * **Task 2:** Create a random input batch (e.g., 3 samples, 4 features).
    * **Task 3:** Perform the forward pass calculation $Z = X \cdot W + B$ and print the shape of the result. Verify it matches your expectation.

* **3. Reading**
    * [Neural Networks and Deep Learning (Ch 1)](http://neuralnetworksanddeeplearning.com/chap1.html) (Section: "The architecture of neural networks").

* **4. Resource(s)**
    * 
    * Video: [3Blue1Brown - But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)

---

## Day 3: Activation & Loss Functions
**Focus:** Adding non-linearity and defining "error" so the network can measure its performance.

* **1. Concepts**
    * **ReLU (Rectified Linear Unit):** $f(z) = max(0, z)$. The standard for hidden layers.
    * **Softmax:** Converts raw scores (logits) into probabilities (summing to 1). Used for the output layer.
    * **Cross-Entropy Loss:** The standard cost function for classification problems (predicting "A" vs "B").

* **2. Coding Practice**
    * **Task 1:** Implement `relu(z)` and `softmax(z)` functions in NumPy. *Tip: For Softmax, subtract the max value from `z` before exponentiating to prevent numerical overflow.*
    * **Task 2:** Implement `cross_entropy(predictions, targets)`.
    * **Task 3:** Pass dummy data through your Day 2 network, apply Softmax, and calculate the Loss against a dummy target.

* **3. Reading**
    * Article: [Understanding Softmax and Cross-Entropy](https://deepnotes.io/softmax-crossentropy) (or similar explanatory blog).

* **4. Resource(s)**
    * 
    * 

---

## Day 4: The Chain Rule (Backpropagation Theory)
**Focus:** Deriving the "gradient" — determining exactly how to adjust weights to reduce error.

* **1. Concepts**
    * **The Gradient:** The vector of partial derivatives pointing in the direction of steepest ascent.
    * **The Chain Rule:** How to calculate $\frac{\partial L}{\partial w}$ by multiplying derivatives layer by layer working backward.
    * Computational Graphs: Visualizing the flow of math.

* **2. Coding Practice**
    * **Task 1:** Watch Andrej Karpathy’s **"Micrograd"** video (first 45 mins). This is crucial.
    * **Task 2:** On paper, draw a simple function like $f(x, y, z) = (x + y) * z$.
    * **Task 3:** Manually calculate the gradients $\frac{df}{dx}, \frac{df}{dy}, \frac{df}{dz}$ for inputs $x=-2, y=5, z=-4$.

* **3. Reading**
    * [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) (Colah's Blog).

* **4. Resource(s)**
    * 
    * Video: [Andrej Karpathy - The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

## Day 5: The Optimizer (Stochastic Gradient Descent)
**Focus:** Using the gradients calculated yesterday to actually update the network's weights.

* **1. Concepts**
    * **Weight Update Rule:** $W_{new} = W_{old} - (\text{Learning Rate} \times \text{Gradient})$.
    * **Learning Rate ($\alpha$):** The step size. Too big = diverge; Too small = slow.
    * **Batch vs. Mini-Batch vs. SGD:** The trade-off between speed and stability.

* **2. Coding Practice**
    * **Task 1:** Create a dummy weight matrix `W` and a dummy gradient `dW`.
    * **Task 2:** Write a loop that updates `W` for 10 steps using a learning rate of 0.1.
    * **Task 3:** Observe how `W` changes. Experiment with a huge learning rate (e.g., 100) and see what happens to the values.

* **3. Reading**
    * [Neural Networks and Deep Learning (Ch 2)](http://neuralnetworksanddeeplearning.com/chap2.html) (Scan the section on SGD).

* **4. Resource(s)**
    * 

---

## Day 6: Project Part I - Building the Engine
**Focus:** Assembling the individual components into a clean, reusable Class structure.

* **1. Concepts**
    * **Object-Oriented Design:** Encapsulating parameters ($W, b$) inside a class.
    * **Forward/Backward API:** The standard interface for all deep learning layers.

* **2. Coding Practice**
    * **Task:** Create a file named `neural_net.py` and implement the class structure:
        ```python
        class NeuralNetwork:
            def __init__(self, input_size, hidden_size, output_size):
                # Initialize weights with He Initialization
                self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
                self.b2 = np.zeros((1, output_size))

            def forward(self, X):
                # Implement Forward Pass & Store values (cache) for backprop
                pass

            def backward(self, X, y, learning_rate):
                # Implement Backward Pass & Update Weights
                pass
        ```

* **3. Reading**
    * Reference: [CS231n Python Numpy Tutorial](https://cs231n.github.io/python-numpy-tutorial/)

* **4. Resource(s)**
    * Python Class Documentation

---

## Day 7: Project Part II - Training on MNIST
**Focus:** The "Hello World" of Deep Learning — training your engine to read handwritten digits.

* **1. Concepts**
    * **Epochs:** One full pass through the entire dataset.
    * **Training Loop:** The standard cycle: Forward -> Loss -> Backward -> Update.
    * **Overfitting:** If training accuracy is 99% but test accuracy is 50%.

* **2. Coding Practice**
    * **Task 1:** Load data using `from sklearn.datasets import load_digits`. Normalize inputs (divide by 16 or 255) and One-Hot Encode the labels.
    * **Task 2:** Write the Training Loop. Iterate for 1000 epochs.
    * **Task 3:** Print the Loss every 100 epochs. It must go down.
    * **Task 4:** Calculate final accuracy on a held-out test set. Target: >85%.

* **3. Reading**
    * None specific (Focus on debugging code).

* **4. Resource(s)**
    * Dataset: 
    *
