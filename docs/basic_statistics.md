# Basic Statistics

## Expectation
In statistics, the **expectation**, or **expected value**, is the long-term average value of a random variable. It represents the mean value you would expect to get if you were to repeat an experiment or process many times. The concept of expectation is fundamental to probability and statistics.

### Calculating Expected Value
The formula for calculating the expected value depends on whether the random variable is discrete or continuous.
* For a **discrete random variable**, which can take on a finite number of distinct values, the expected value is calculated by multiplying each possible value by its probability and then summing these products.
* For a **continuous random variable**, which can take on any value within a given range, the expected value is calculated by integrating the product of the variable and its probability density function over the entire range of possible values.

$`\begin{equation}
E_{x \sim P} [f(x)] = 
  \begin{cases}
    {\sum_x P(x) f(x) discrete P} \\
    \\
    {\int_x P(x) f(x) dx continuous P}
  \end{cases}
\end{equation}`$

Some short forms of $`E_{x \sim P} [f(x)]`$ are
* $`E_P [f(x)]`$
* $`E [f(x)]`$
* $`E_P [f]`$
* $`E [f]`$

### Linearity of expectation
* $`E[f(x) + g(x)] = E[f(x)] + E[g(x)]`$
* $`E[\alpha f(x)] = \alpha E[f(x)]`$

## Mean and Variance
### Mean: Average value of the distribution
$`\mu_x = E_{x \sim P} [x]`$
### Variance: The spread of the values in a distribution
$`\sigma_x^2 = Var_{x \sim P} [x] = E_{x \sim P} [(x - \mu_x)^2]`$

### Gaussian Distribution

### Discrete distribution

### Continuous distribution

## Sampling
x ~ P

### Bias in samples
Samples are always biased
For infinite samples: empirical distribution = data generating distribution

## Statistical Models
Statistical models are functions that map certain inputs to certain outputs.

### Regression model

$`f_\theta : \mathbb{R}^n \to \mathbb{R}^d `$

\\Parameterized

### Classification model

$`f_\theta : \mathbb{R}^n \to P(X), P(X) \subset \mathbb{R}^d `$

\\Parameterized

### Statistical Summary
$`f_\theta : X \to Y`$
* X: input space
* Y: output space
* \theta: model parameters

\\In machine learning
* Goal: find the optimal parameter **\theta**
* How: learn from **data**

## Data
### Unlabeled data
D = (x1, x2, ... xn) where xi ~ P..

### Labeled data
D = {(x1, l1), (x2, l2), ... (xn, ln))} where xi ~ P and li ...


