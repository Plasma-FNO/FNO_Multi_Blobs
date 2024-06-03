Plasma Surrogate Modelling using Fourier Neural Operators
This repository contains the code and experiments for the paper "Plasma Surrogate Modelling using Fourier Neural Operators" by Vignesh Gopakumar et al., published in Nuclear Fusion, 2024.
Overview
The paper demonstrates the use of Fourier Neural Operators (FNO) as surrogate models for predicting plasma evolution in both simulation and experimental domains.

Key highlights include:

1. A modified version of the FNO capable of handling multiple variables that govern a family of PDEs, allowing for learning and encoding the mutual dependence of variables.
2. Demonstration of the FNO's utility in surrogate modelling the evolution of plasma across a range of reduced magnetohydrodynamic (MHD) cases with increasing complexity.
3. Application of FNOs to predict plasma evolution on real-world experimental data observed by cameras positioned within the MAST Tokamak.
4. Extensive analysis of the FNO's features, including ablation studies evaluating the impact of various hyperparameters and training regimes.
5. Comparison of the performance of individual FNOs with the multi-variable FNO.

The FNO offers a viable alternative for surrogate modelling, as it is quick to train and infer, requires fewer data points, and can perform zero-shot super-resolution while achieving high-fidelity solutions.
