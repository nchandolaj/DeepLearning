# Deep Learning: What is a Forward Pass?

In Deep Learning, a **forward pass** (also known as **forward propagation**) is the process where input data travels through the layers of a neural network to produce an output or "prediction."

Think of it as the network's "inference" phase—it is the step where the model actually looks at the data and makes a guess before any learning or correction happens.

## How it Works: Step-by-Step
During a forward pass, data flows in one direction: from the **Input Layer**, through one or more **Hidden Layers**, to the **Output Layer**.

At every single neuron in the network, two main mathematical operations occur:

1.  **The Weighted Sum ($z$):** Each input is multiplied by a "weight" (importance) and a "bias" (offset) is added.
    $$z = \sum (weight \times input) + bias$$
2.  **The Activation Function ($a$):** This result ($z$) is passed through a non-linear function like **ReLU** or **Sigmoid**. This allows the network to learn complex patterns rather than just simple linear relationships.
    $$a = \sigma(z)$$


## The Forward Pass in the Training Cycle
In the context of training a model, the forward pass is only the first half of the story. The full cycle looks like this:

| Step | Phase | Action |
| :--- | :--- | :--- |
| **1** | **Forward Pass** | Input data goes in $\rightarrow$ Prediction comes out. |
| **2** | **Loss Calculation** | The prediction is compared to the "correct" answer to see how wrong it was (the "Loss"). |
| **3** | **Backward Pass** | The error is sent back through the network to calculate how much each weight contributed to the mistake. |
| **4** | **Optimizer Step** | Weights are updated slightly to reduce the error for the next time. |

## Key Takeaways
* **Direction:** It always moves from left to right (Input $\rightarrow$ Output).
* **Purpose:** To generate a prediction or calculate the current "loss."
* **Inference vs. Training:** When you use a finished model (like ChatGPT or FaceID), it *only* performs a forward pass. The backward pass is only used during the training phase to help the model learn.

---

# APPENDIX

## Forward propagation in Neural Network

```mermaid
graph LR
    subgraph Input Layer
        i1((Input 1))
        i2((Input 2))
        i3((Input 3))
    end

    subgraph Hidden Layer
        h1((Hidden 1))
        h2((Hidden 2))
        h3((Hidden 3))
        h4((Hidden 4))
        h5((Hidden 5))
    end

    subgraph Output Layer
        o1((Output 1))
    end

    %% Connect input layer to hidden layer
    i1 --> h1
    i1 --> h2
    i1 --> h3
    i1 --> h4
    i1 --> h5

    i2 --> h1
    i2 --> h2
    i2 --> h3
    i2 --> h4
    i2 --> h5

    i3 --> h1
    i3 --> h2
    i3 --> h3
    i3 --> h4
    i3 --> h5

    %% Connect hidden layer to output layer
    h1 --> o1
    h2 --> o1
    h3 --> o1
    h4 --> o1
    h5 --> o1

    style i1 fill:#f9f,stroke:#333,stroke-width:2px
    style i2 fill:#f9f,stroke:#333,stroke-width:2px
    style i3 fill:#f9f,stroke:#333,stroke-width:2px

    style h1 fill:#ccf,stroke:#333,stroke-width:2px
    style h2 fill:#ccf,stroke:#333,stroke-width:2px
    style h3 fill:#ccf,stroke:#333,stroke-width:2px
    style h4 fill:#ccf,stroke:#333,stroke-width:2px
    style h5 fill:#ccf,stroke:#333,stroke-width:2px

    style o1 fill:#cfc,stroke:#333,stroke-width:2px

    linkStyle 0,1,2,3,4,5,6,7,8,9,10,11 stroke-width:2px,fill:none,stroke:grey;
```

## Perception

```mermaid
flowchart LR
    x1((x₁)) -->|w₁| sum((Σ))
    x2((x₂)) -->|w₂| sum
    x3((x₃)) -->|w₃| sum

    b((bias)) --> sum

    sum --> act((Activation<br/>Function))
    act --> y((Output y))

```

### What this represents
* **x₁, x₂, x₃:** Input features
* **w₁, w₂, w₃:** Weights (shown on edges)
* **bias:** Bias term added to the weighted sum
* **Σ:** Linear combination

z = $∑w_i​x_i$ + b

* **Activation:** Applies a function (e.g., step, sigmoid, ReLU)
* **Output y:** Final perceptron output

