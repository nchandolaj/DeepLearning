# Understanding Tensors: Beyond the Matrix

While a **Matrix** is a 2D grid of numbers, a **Tensor** is the umbrella term for any $n$-dimensional array. In Deep Learning frameworks like PyTorch or TensorFlow, almost all data is stored as a Tensor.


## 1. The Dimensional Hierarchy (Ranks)
The "Rank" of a tensor refers to the number of dimensions it has.

| Rank | Name | Visual Description | Example Use Case |
| :--- | :--- | :--- | :--- |
| **0** | **Scalar** | A single number | The "Loss" value or a Learning Rate. |
| **1** | **Vector** | A 1D line of numbers | A single bias vector $B$. |
| **2** | **Matrix** | A 2D grid (Rows $\times$ Cols) | A single layer of weights $W$. |
| **3** | **3D Tensor** | A "cube" of numbers | A sequence of text (Batch, Sequence, Features). |
| **4** | **4D Tensor** | A "row" of cubes | A batch of color images (Batch, Height, Width, Channels). |


## 2. Why Tensors are Essential for Imagery
A standard color image is a **3D Tensor**. It has a height and width (the pixels), but it also has a "depth" of 3 (Red, Green, and Blue channels).

* **Single Image:** $(Height, Width, 3)$
* **Batch of Images:** To process multiple images at once using vectorization, we add a 4th dimension: $(Batch\_Size, Height, Width, 3)$.

When you hear about "Tensor Operations," it usually refers to performing math (like addition or multiplication) across all these dimensions simultaneously.


## 3. Key Tensor Attributes
When debugging Deep Learning code, you will constantly check three things:

1.  **Shape:** The size of each dimension (e.g., `(32, 224, 224, 3)`).
2.  **Rank:** The number of dimensions (for the shape above, the rank is 4).
3.  **Data Type (dtype):** Tensors usually hold 32-bit floats (`float32`). If you try to multiply a `float32` tensor by an `int64` tensor, the code will usually crash.


## 4. Tensor Reshaping: The "Plasticity" of Data
One of the most common matrix operations is **Reshaping**. 
Imagine a $(28 \times 28)$ grayscale image. To pass it into a standard "Dense" layer, we must "flatten" it into a 1D Vector of $784$ numbers.

* **Original:** $(28, 28)$
* **Reshaped:** $(784,)$

> **Important Rule:** You can reshape a tensor into any shape as long as the total number of elements stays exactly the same ($28 \times 28 = 784$).

---

## Tensor Cheat Sheet: PyTorch, TensorFlow, and JAX
To implement the concepts we discussed (Vectorization, Shapes, and the Layer Equation), you’ll need to know how to handle Tensors in the major frameworks.

While they all perform similar math, their "personalities" differ: 
* **PyTorch** is object-oriented and Pythonic,
* **TensorFlow** is built for large-scale production, and
* **JAX** is a functional, high-performance library that feels like NumPy on steroids.

Below are simple code examples for declaring and instantiating tensors in the three most common frameworks.

### 1. PyTorch (`torch.Tensor`)
PyTorch is currently the most popular framework for research. Its API is very similar to standard Python.

```python
import torch

# From a Python list
x = torch.tensor([[1, 2], [3, 4]])

# Common initializers
zeros = torch.zeros((2, 3))       # (Batch_Size, Features)
ones  = torch.ones((2, 3))
rand  = torch.rand((2, 3))       # Uniform distribution [0, 1)

# Attributes
print(f"Shape: {x.shape}")        # torch.Size([2, 2])
print(f"Device: {x.device}")      # cpu or cuda:0
```

### 2. TensorFlow (`tf.Tensor`)
TensorFlow (often used with Keras) is a powerhouse for production and mobile deployment.

```python
import tensorflow as tf

# From a Python list (Constants are immutable)
x = tf.constant([[1, 2], [3, 4]])

# Trainable parameters (Mutable tensors for weights/biases)
w = tf.Variable(tf.random.normal((784, 512)))

# Common initializers
zeros = tf.zeros((2, 3))
ones  = tf.ones((2, 3))
rand  = tf.random.uniform((2, 3))

# Attributes
print(f"Shape: {x.shape}")        # (2, 2)
print(f"Dtype: {x.dtype}")        # <dtype: 'int32'>
```

### 3. JAX (`jax.Array`)
JAX is used by DeepMind and Google Research. It uses jax.numpy (jnp) to mirror the NumPy API almost perfectly.

```python
import jax
import jax.numpy as jnp

# From a Python list
x = jnp.array([[1, 2], [3, 4]])

# Common initializers
zeros = jnp.zeros((2, 3))
ones  = jnp.ones((2, 3))

# Randomness in JAX is explicit (requires a 'key')
key = jax.random.PRNGKey(0)
rand = jax.random.uniform(key, (2, 3))

# Attributes
print(f"Shape: {x.shape}")        # (2, 2)
```

### Summary Comparison
| Feature | PyTorch | TensorFlow | JAX |
| :-- | :-- | :-- | :-- |
| Main Object | torch.Tensor | tf.Tensor | jax.Array |
| Mutability | Mutable | Immutable (mostly) | Strictly Immutable |
| Philosophy | Object-Oriented | Production-First | Functional Programming |
| Device Placement | Manual (.to('cuda')) | Automatic (mostly) | Automatic / Explicit |

### Pro-Tip: The Layer Equation across frameworks
Despite these syntax differences, the **Layer Equation** $Z = XW + B$ looks nearly identical in all three:
* **PyTorch:** Z = X @ W + B
* **TensorFlow:** Z = tf.matmul(X, W) + B
* **JAX:** Z = jnp.dot(X, W) + B
  
