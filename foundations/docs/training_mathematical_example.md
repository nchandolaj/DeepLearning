# Neural Network: A Concrete Mathematical Example

Let’s walk through a single "learning step" for a tiny neural network. We will use a computational graph approach to find out how to update a weight to reduce our error.

## 1. The Setup

Imagine a single neuron with one input ($x$), one weight ($w$), and no bias. We use a **Squared Error** loss function.

* **Input ($x$):** $2.0$
* **Target ($y$):** $1.0$ (The "correct" answer we want)
* **Initial Weight ($w$):** $3.0$

**The Math:**
1.  **Prediction ($p$):** $x \times w$
2.  **Loss ($L$):** $(p - y)^2$

## 2. The Forward Pass

First, we calculate the current output and how "wrong" we are.

1.  **Calculate $p$:** $2.0 \times 3.0 = \mathbf{6.0}$
2.  **Calculate $L$:** $(6.0 - 1.0)^2 = 5.0^2 = \mathbf{25.0}$

Our current Loss is **25.0**. Our goal is to change $w$ to make this number smaller.

## 3. The Backward Pass (Calculating Gradients)

To know how to change $w$, we need the derivative $\frac{\partial L}{\partial w}$. Using the **Chain Rule**, we break this into two steps:

### Step A: How much does the Loss change relative to the Prediction? ($\frac{\partial L}{\partial p}$)

Since $L = (p - y)^2$, the derivative (using the power rule) is $2(p - y)$.
* $\frac{\partial L}{\partial p} = 2(6.0 - 1.0) = \mathbf{10.0}$

### Step B: How much does the Prediction change relative to the Weight? ($\frac{\partial p}{\partial w}$)

Since $p = x \times w$, the derivative with respect to $w$ is simply $x$.

* $\frac{\partial p}{\partial w} = x = \mathbf{2.0}$

### Step C: The Chain Rule

Now we multiply them together to find the final gradient:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \times \frac{\partial p}{\partial w}$$
$$\frac{\partial L}{\partial w} = 10.0 \times 2.0 = \mathbf{20.0}$$

## 4. The Update (Gradient Descent)

The gradient is **20.0**. This tells us that if we increase $w$ by a tiny bit, the Loss will increase 20 times faster. Since we want to **decrease** the Loss, we move in the opposite direction.

If our **Learning Rate ($\eta$)** is $0.01$:

$$New\ Weight = w - (\eta \times \frac{\partial L}{\partial w})$$
$$New\ Weight = 3.0 - (0.01 \times 20.0) = \mathbf{2.8}$$

## 5. Verification
Let's run a **Forward Pass** again with our new weight ($2.8$):
1.  **New Prediction ($p$):** $2.0 \times 2.8 = \mathbf{5.6}$
2.  **New Loss ($L$):** $(5.6 - 1.0)^2 = 4.6^2 = \mathbf{21.16}$

**Result:** The Loss dropped from **25.0** to **21.16**. The model just learned!

## Summary Table

| Variable | Before Update | After Update | Change |
| :--- | :--- | :--- | :--- |
| **Weight ($w$)** | 3.0 | 2.8 | -0.2 |
| **Prediction ($p$)** | 6.0 | 5.6 | -0.4 |
| **Loss ($L$)** | 25.0 | 21.16 | **-3.84 (Improved!)** |
