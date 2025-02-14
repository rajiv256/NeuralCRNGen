# Neural CRNs

This repository contains code for running the simulations presented in the paper [Neural CRNs: A Natural Implementation of Learning in Chemical Reaction Networks](https://doi.org/10.48550/arXiv.2409.00034)

## Branches for different experiments

- [Linear Regression](https://github.com/rajiv256/NeuralCRNGen/tree/linear_regression) ($`y = k_1 x_1+ k_2 x_2 + k_0`$)
- [Nonlinear Regression](https://github.com/rajiv256/NeuralCRNGen/tree/nonlinear_regression_z2_dotprod) ($`y = x_1 x_2 + x_2^2`$, $`f_\theta = \theta \odot x + \beta - {z}^2`$)
- [Nonlinear Regression](https://github.com/rajiv256/NeuralCRNGen/tree/nonlinear_regression_z2_dotprod_sinxx2) ($`y = \sin(x_1) + x_2^2`$, $`f_\theta = \theta \odot x + \beta - {z}^2`$)
- [Nonlinear Classification with a cubic polynomial system](https://github.com/rajiv256/NeuralCRNGen/tree/nonlinear_z3) ($`f_\theta = \theta x - {z}^3`$)
- [Nonlinear Classification with a quadratic polynomial system](https://github.com/rajiv256/NeuralCRNGen/tree/nonlinear_z3_linapprox) ($`f_\theta = \theta x - \alpha {z}^2`$)
