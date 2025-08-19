# PINNs for Diffusion Problems

This repository contains two case studies of solving diffusion-type PDEs using **Physics-Informed Neural Networks (PINNs)** with **DeepXDE** (TensorFlow backend) and a custom **PyTorch PINN**.

## Contents

- `Fokker–Planck equation- Diffusion problem.ipynb`  
  PINN solution of a diffusion equation in **log-γ space** with analytic comparison.

- `simple_diffusion_pinns.ipynb`  
  1D Gaussian diffusion solved using **DeepXDE** and a **custom PyTorch PINN**.


## Case 1 — Diffusion in log-γ Space (DeepXDE)

We transform $y$ via $h = \log y$ and work in $(g,\tau)$ where $g=\ln \gamma$.

### PDE (implemented in the notebook)

$$
\frac{\partial h}{\partial \tau}
=\frac{\partial^2 h}{\partial g^2}
+
\left(\frac{\partial h}{\partial g}\right)^{2}
+
\frac{\partial h}{\partial g}
$$


### Domain and Conditions
- Domain: $g \in [\ln 1, \ln 10^6],\; \tau \in [1,3]$  
- BCs: Zero-flux (Neumann) at both $g$ boundaries  
- IC: Analytic solution at $\tau=1$

### Analytic solution for $y$
$$
y(g,\tau) =
\frac{1}{\gamma \,\sqrt{4\pi \tau}}
\exp\!\left(
-\frac{(\ln(\gamma_0/\gamma)+\tau)^2}{4\tau}
\right),
\qquad
\gamma = e^g,\;\; \gamma_0 = 100 .
$$

We train a PINN on $h=\log y$ and compare $y_{\text{PINN}} = \exp(h)$ with $y_{\text{analytic}}$.

**Result:** $L_1$ error ≈ **6.3%**, showing strong agreement.

## Case 2 — Gaussian Diffusion (DeepXDE & PyTorch)

### PDE
$$
\frac{\partial u}{\partial t}
= D \,\frac{\partial^2 u}{\partial x^2},
\qquad
x\in[0,1],\; t\in[0,1],\; D=0.1 .
$$

### Initial Condition (Gaussian)
$$
u(x,0) =
\exp\!\left(-\frac{(x-x_0)^2}{2\sigma^2}\right),
\qquad x_0=0.5,\; \sigma=0.1 .
$$

### Boundary Conditions
$$
u(0,t) = u(1,t) = 0.
$$

### Analytic solution
$$
u(x,t) =
\frac{1}{\sqrt{1+2Dt}}
\exp\!\left(
-\frac{(x-x_0)^2}{2\sigma^2(1+2Dt)}
\right).
$$


### DeepXDE (TF) version
- Small FNN: `[2] → [5,5,5] → [1]`, `tanh` activations.  
- Trains but with **high relative error (~0.62)** → under-fit.

### PyTorch PINN version
- Manual residual losses for PDE, IC, BC.  
- MLP with `tanh`, Adam optimizer.  
- **Good agreement** with analytic solution, much lower error than DeepXDE setup.

## How to Run

```bash
# Install dependencies
pip install deepxde tensorflow torch matplotlib numpy jupyter

# Launch notebooks
jupyter notebook "Fokker–Planck equation- Diffusion problem.ipynb"
jupyter notebook simple_diffusion_pinns.ipynb
