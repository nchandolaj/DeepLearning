# Deep Learning: What is a Backward Pass?

The **backward pass**, commonly known as **backpropagation**, is the process in deep learning where a neural network calculates the **gradients** of the loss function with respect to its weights and biases. 

If the forward pass is the network "making a guess," the backward pass is the network "learning from its mistakes." It is the mathematical engine that allows a model to improve over time.


## ⚙️ How the Backward Pass Works

The backward pass moves in the opposite direction of the data flow (from the output layer back toward the input layer). It follows three main steps:


### 1. The Error Signal
It begins at the very end of the forward pass. Once the **Loss Function** calculates the error (the difference between the prediction $\hat{y}$ and the target $y$), the backward pass computes the gradient of that loss. This serves as the "starting signal" for the rest of the chain.


### 2. The Chain Rule
The core of the backward pass is the **Chain Rule** from calculus. Because a neural network is a composition of many functions (layers), the chain rule allows us to calculate how a weight in an early layer affects the final loss by multiplying local derivatives.

For a specific weight $w$ in a hidden layer, the gradient is calculated as:

$$\frac{\partial \text{Loss}}{\partial w} = \frac{\partial \text{Loss}}{\partial \text{Output}} \times \frac{\partial \text{Output}}{\partial \text{Layer}} \times \frac{\partial \text{Layer}}{\partial w}$$


### 3. Gradient Calculation
As the process moves backward through each layer, it calculates two things:
* **The Gradient for Parameters:** How to change the weights and biases of *this* layer to reduce the loss.
* **The Gradient for the Previous Layer:** The error signal that needs to be passed further back to the layers that came before it.


## 🎯 Why is it "Backward"?

It is called a "backward" pass because of **computational efficiency**. 

If we tried to calculate the influence of every weight starting from the front (input), we would end up repeating the same massive calculations millions of times for every single parameter. By starting at the **output** and moving backward, we can calculate the gradient for one layer and **reuse** that result to calculate the gradients for the layer before it. This "memoization" makes training deep networks feasible.


## 🚀 The Result: Optimization

The backward pass itself does **not** change the weights; it only calculates the "map" or "direction" of how they *should* change. 

Once the backward pass is finished and all gradients are stored, an **Optimizer** (like SGD or Adam) takes those gradients and performs the actual update:
$$\text{Weight} = \text{Weight} - (\text{Learning Rate} \times \text{Gradient})$$

**Summary Table**
| Feature | Forward Pass | Backward Pass |
| :--- | :--- | :--- |
| **Direction** | Input $\rightarrow$ Output | Output $\rightarrow$ Input |
| **Goal** | Make a prediction | Calculate gradients (error attribution) |
| **Key Operation** | Matrix multiplication & Activation | Chain Rule & Derivatives |
| **Output** | Prediction ($\hat{y}$) and Loss ($L$) | Gradients ($\nabla W, \nabla b$) |
