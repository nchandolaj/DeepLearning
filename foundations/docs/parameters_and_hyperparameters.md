# Parameters and Hyperparameters

In the context of Deep Networks, the distinction between **Parameters** and **Hyperparameters** is crucial. Think of the network as a machine you are building and training.

* **Parameters** are the internal parts of the machine that move and adjust themselves automatically.
* **Hyperparameters** are the settings on the control panel that *you* (the human) must set before you turn the machine on.


## 1. What are Parameters? (The "Learned Knowledge")
Parameters are the internal variables that the model learns from the training data. They are the values that change during the "training" process via backpropagation.

* **Definition:** Variables optimized by the algorithm to minimize the loss function.
* **Who sets them?** The Model (automatically).
* **When are they set?** During the training phase (they start random and get refined).
* **Examples:**
    * **Weights ($W$):** The strength of the connection between two neurons.
    * **Biases ($b$):** The threshold or offset for a neuron.

> **Analogy:** If a neural network is a student studying for an exam, the **Parameters** are the connections in the student's brain (synapses) that strengthen as they learn the material.


## 2. What are Hyperparameters? (The "Configuration")
Hyperparameters are the external configurations that you define *before* training begins. They control *how* the model learns and what the model's structure looks like. The model cannot change these on its own.

* **Definition:** Settings that govern the training process and the architecture of the model.
* **Who sets them?** You (the Data Scientist/Engineer).
* **When are they set?** Before training starts.
* **Examples:**
    * **Learning Rate:** How big of a step the model takes during optimization (e.g., 0.001 vs 0.01).
    * **Number of Epochs:** How many times the model sees the entire dataset.
    * **Batch Size:** How many data examples the model processes at once.
    * **Architecture:** Number of layers, number of neurons per layer, type of activation function.

> **Analogy:** In the student analogy, the **Hyperparameters** are the study environment: "How many hours will they study?" (Epochs), "Will they study alone or in a group?" (Batch Size), "How fast will they try to read?" (Learning Rate).


## 3. Comparison Table

| Feature | Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Origin** | Learned from data | Set manually by human (or tuning algos) |
| **Status** | Internal to the model | External to the model |
| **Final State** | Saved as part of the trained model file | Not saved in the model file (only used to create it) |
| **Example** | Weights ($W$), Biases ($b$) | Learning Rate ($\alpha$), # of Layers, Dropout rate |
| **How to optimize** | Backpropagation / Gradient Descent | Grid Search, Random Search, Bayesian Optimization |


## 4. Why the distinction matters
If you have a "bad" model, you need to know which knob to turn:
1.  **Underfitting (Model is too dumb):** You might need to change the **Hyperparameters** (add more layers, train longer).
2.  **Overfitting (Model memorizes data):** You might need to change **Hyperparameters** (add Dropout, increase regularization) or get more data to help the **Parameters** generalize better.

The process of finding the best hyperparameters is called **"Hyperparameter Tuning."** It is often trial-and-error, whereas finding the best parameters is purely mathematical (Calculus).



