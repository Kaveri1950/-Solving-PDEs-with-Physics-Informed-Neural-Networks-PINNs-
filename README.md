# Solving PDEs with Physics Informed Neural Networks (PINNs)


# PINNs for Diffusion Problems

This project contains two case studies where we apply **Physics-Informed Neural Networks (PINNs)** to solve diffusion-type equations using **DeepXDE (TensorFlow backend)** and a **custom PyTorch implementation**.

## Project Structure

- `01_log_gamma_diffusion.ipynb` → Diffusion in **log-γ space** (DeepXDE)  
- `02_gaussian_diffusion.ipynb` → Gaussian diffusion (DeepXDE + PyTorch)  

## Case 1: Diffusion in log-γ Space (DeepXDE)

We solve a PDE in transformed **logarithmic γ-space**:

\[
\frac{\partial f}{\partial \tau} = \frac{\partial}{\partial g}\left( e^{2g} \frac{\partial f}{\partial g} \right),
\quad g = \ln \gamma
\]

- **Domain:** \( g \in [\ln 1, \ln 10^6], \; \tau \in [1, 3] \)  
- **Initial condition:** Delta-function like peak at \(\gamma_0 = 100\)  
- **Boundary condition:** Zero-flux (Neumann) at domain boundaries  

PINN solution matches the **analytical Green’s function solution** with ~**6.3% relative error**.

## Case 2: Gaussian Diffusion

We solve the **1D heat/diffusion equation**:

\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}, 
\quad x \in [-5,5], \; t \in [0,1]
\]

- **Diffusion coefficient:** \( D = 1.0 \)  
- **Initial condition:**

\[
u(x,0) = \exp\!\left(-\frac{x^2}{2\sigma^2}\right), \quad \sigma = 0.5
\]

- **Boundary condition:** \(u(-5,t) = u(5,t) = 0\) (Dirichlet)  
- **Analytical solution:**

\[
u(x,t) = \frac{1}{\sqrt{1+2Dt}} 
\exp\!\left(-\frac{x^2}{2\sigma^2(1+2Dt)}\right)
\]

### Implementation A: DeepXDE
- Used `FNN` with tanh activations.  
- Accuracy was **poor (~62% relative error)** with a shallow network.  
- Needs larger architecture & better collocation sampling.

### Implementation B: PyTorch PINN
- PDE residual, IC, and BC losses coded manually.  
- Network: fully connected MLP with tanh activations.  
- Trained with Adam optimizer.  
- Achieved **good match to analytical solution** — Gaussian broadens as expected.


## Results Summary

| Case | Framework | Accuracy | Notes |
|------|-----------|----------|-------|
| Log-γ diffusion | DeepXDE | ~6.3% error | Stable & accurate |
| Gaussian diffusion | DeepXDE | ~62% error | Too shallow NN |
| Gaussian diffusion | PyTorch PINN | Low error | Good agreement |


## How to Run

### Install dependencies
```bash
pip install deepxde tensorflow torch matplotlib numpy
