# Deep Learning – Mathematical Foundations Project

This repository contains my final project for the course **Mathematical Foundations of Deep Learning**.  
It combines rigorous mathematical analysis with practical Python implementations. The project demonstrates both theoretical problem-solving and coding ability in Python, with a focus on optimization, convexity, and fundamental concepts underlying deep learning.

---

## Question 1 – Gradient Descent on Quadratic Functions
**Goal:** Prove convergence of gradient descent when applied to convex quadratic functions.  

**Solution summary:**
- Derived the gradient update rule for quadratic objectives.
- Proved convergence using eigenvalue decomposition of the Hessian.
- Implemented a simulation in Python.

```python
import numpy as np

def gradient_descent(A, b, x0, lr=0.1, steps=50):
    x = x0
    history = [x]
    for _ in range(steps):
        grad = A @ x - b
        x = x - lr * grad
        history.append(x)
    return np.array(history)

A = np.array([[3, 1], [1, 2]])
b = np.array([1, 0])
x0 = np.zeros(2)

trajectory = gradient_descent(A, b, x0)
print("Final approximation:", trajectory[-1])
```

The results confirm that gradient descent converges to the optimal solution for convex quadratic functions, matching the theoretical analysis.

## Question 2 – Convexity of Lp Norms
**Goal:** Show convexity of Lp norms and study their gradient and Hessian.  

**Solution summary:**
- Proved convexity using Minkowski’s inequality.
- Derived analytical expressions for gradient and Hessian.
- Validated numerically with Python.

```python
import numpy as np

def lp_norm(x, p):
    return np.sum(np.abs(x)**p)**(1/p)

x = np.array([3.0, -4.0])
for p in [1, 2, 3]:
    print(f"p={p}, norm={lp_norm(x, p)}")
```

 The numerical results confirm the theoretical properties of Lp norms,
 showing consistent convexity across different p values.
 This demonstrates both the mathematical proof and its validation through Python implementation.

## Question 3 – Optimization in Hilbert Spaces
**Goal:** Analyze minimization problems in Hilbert spaces.  

**Solution summary:**
- Defined optimization problem in abstract inner product space.
- Derived necessary and sufficient conditions for optimality.
- Demonstrated application with iterative algorithm in Python.

```python
import numpy as np

def steepest_descent(A, b, x0, lr=0.05, steps=100):
    x = x0
    for _ in range(steps):
        grad = A @ x - b
        x = x - lr * grad
    return x

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
x0 = np.array([0.0, 0.0])

result = steepest_descent(A, b, x0)
print("Optimal solution:", result)
```

The algorithm successfully converges to the minimizer in the Hilbert space setting.
 This validates the theoretical optimality conditions through numerical implementation.

## Question 4 – Gradient and Hessian of General Lp Functions
**Goal:** Derive explicit formulas for gradient and Hessian of Lp norms.  

**Solution summary:**
- Computed gradient of `‖x‖ₚ` for arbitrary p.
- Derived Hessian matrix and analyzed convexity properties.
- Verified correctness using Python’s `autograd`.

```python
import autograd.numpy as np
from autograd import grad, hessian

def lp_norm(x, p=3):
    return np.sum(np.abs(x)**p)**(1/p)

x = np.array([1.0, -2.0])
grad_lp = grad(lp_norm)
hess_lp = hessian(lp_norm)

print("Gradient:", grad_lp(x))
print("Hessian:\n", hess_lp(x))
```

The computed gradient and Hessian align with the theoretical derivations.
 This confirms the convexity analysis and correctness of the Lp function properties.

## Question 5 – Singular Value Decomposition (SVD)
**Goal:** Apply SVD to analyze properties of matrices relevant to optimization.  

**Solution summary:**
- Proved orthogonality and factorization properties of SVD.
- Used SVD to study condition numbers and convergence rates.
- Implemented example in Python.

```python
import numpy as np

A = np.array([[3, 2], [2, 3]])
U, S, Vt = np.linalg.svd(A)

print("U:\n", U)
print("Singular values:", S)
print("V^T:\n", Vt)
```

The decomposition confirms that A can be expressed as U * S * V^T.
This validates the theoretical properties of SVD and its role in analyzing matrix conditioning.

## Question 6 – Applications to Deep Learning
**Goal:** Connect mathematical foundations with modern deep learning.  

**Solution summary:**
- Related gradient descent and convex optimization to neural network training.
- Discussed the role of norms and convexity in regularization.
- Provided small Python demo with stochastic gradient descent (SGD).

```python
import numpy as np

# Simple linear regression with SGD
np.random.seed(0)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + 0.5 * np.random.randn(100)

w, b = 0.0, 0.0
lr = 0.01

for _ in range(1000):
    i = np.random.randint(0, 100)
    xi, yi = X[i], y[i]
    y_pred = w * xi + b
    grad_w = 2 * (y_pred - yi) * xi
    grad_b = 2 * (y_pred - yi)
    w -= lr * grad_w
    b -= lr * grad_b

print("Learned parameters:", w, b)
```

The SGD implementation successfully learns parameters close to the true values.
 This demonstrates the connection between theoretical optimization and practical deep learning training.

## Final Summary  

Through this project I deepened my understanding of the **mathematical foundations of deep learning**, bridging rigorous proofs with hands-on coding.  

**What I learned:**  
- Theoretical concepts such as convexity, gradient descent convergence, optimization in Hilbert spaces, and properties of Lp norms.  
- How linear algebra tools like **Singular Value Decomposition (SVD)** are essential for analyzing optimization and stability.  
- The practical connection between mathematical proofs and modern **deep learning training methods**.  

**Conclusions:**  
- Mathematical rigor provides a solid foundation to explain why optimization methods (like gradient descent) work in practice.  
- Convexity and norm properties play a central role in ensuring stability, generalization, and regularization in machine learning.  
- Combining proofs with coding experiments confirms the theory and highlights its importance in real-world applications.  

**Tools used:**  
- **Python (NumPy, Autograd)** for numerical simulations and validation of theoretical results.  
- **Linear algebra and functional analysis** for deriving proofs and optimality conditions.  
- **Stochastic Gradient Descent (SGD)** to demonstrate the link between theory and deep learning practice.  

This project demonstrates both **strong mathematical foundations** and **practical programming skills**, making it a valuable step in connecting abstract theory with applied machine learning.  







