## Probability

**P(X = a)**  
* 'X' is a **random variable** and 'a' is a **specific value** (the outcome itself)
* '(X = a)' is an **event** in which the **random valiable** equals a **specific value**
* 'P' is the **probability** of an event happening

**A Coin Toss**:
* The **specific value** may be 'heads' (or 'tails' if you would like). The **random variable** is the 'coin landing on'.
* The **event** is the 'toss' of the coin. In this case, the 'coin landing on' it's 'heads'.
* The **probability** of the **event** happening. That is, the probability of the toss resulting in the coin landing on it's 'heads'.

**Probability Density**
* $P(Y = \alpha)$
  - Not defined
* Cumulative probability: $P(\alpha_1 \leq Y < \alpha_2)$
* Probability Density: $p(Y = \alpha)$
  - $p(Y = \alpha) = \frac{P(\alpha - \epsilon \leq Y < \alpha + \epsilon)}{2\epsilon}$

**P(x) = P(X = x)**  Probability of a discrete event
**P(x) = p(X = x)**  Probability density of a continuous event

**Properties of P(x)**
* Non-negativity: $0 \leq P(x)$
* Boundless (discrete only):  $0 \leq P(x) \leq 1$
* Summation (discrete): $E_p[1] = \sum_x P(x) = 1$
* Summation (continuous): $E_p[1] = \int P(x) dx = 1$

**What is P in P(x)?**
* 'P' is a A function
* In the discrete case, $P : (c_1, c_2, ..., c_n) \rightarrow [0,1]$
* In the continuous case, $P : \mathbb{R} \to \mathbb{R}$
* 'P' is called a **probability distribution**

**Three types of distribution**
1. **Data generating**: It lives in the real world, e.g., the weather data. 
2. **Empirical**: Samples of the real data that we observe, e.g., weather measurements of the samples from the weather. 
3. **Model**: If 
