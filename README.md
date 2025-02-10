# Preliminaries 

Make sure you have `Julia v1.11.0+` installed. 

# Execution

To execute, navigate to the root directory `NeuralCRNGen/` and execute

```
julia julia/neuralcrn_dotprod.jl
```

_Note: If you don't have certain packages installed, execution might throw an error. In that case, uncomment the `Pkg.add("DifferentialEquations")` (for example) at the top of the file._

This command executes the `neuralcrn()` function. One important parameter that needs to be provided is the `output_dir`. It takes the value of a  folder name and creates the folder at `julia/<output_dir>`. All the image files and the output files will be populated into this folder. 

# Useful Files and Directories

- The CRNs could be found in `julia/reactions_dotprod.jl`.
- The function for creating the dataset could be found in `julia/datasets.jl`.
- The util functions for creating the plots could be found in `julia/myplots.jl`.
- Other general utils functions can be found in `julia/utils.jl`.

# Important functions

- The code for training the Neural CRN model is in `crn_main(...)` function. 

# First-order approximation of gradients

A simpler backpropagation CRN could be obtained by removing the adjoint and hidden state backpropagation CRNs from the feedback phase. The results of this experiments are as follows. 

To execute, navigate to the root directory `NeuralCRNGen/` and execute

```
julia julia/neuralcrn_dotprod_simpler.jl
```

The CRN file for this could be found at `julia/reactions_dotprod_simpler.jl`. Notice that the adjoint and hidden state backpropagations would be commented out in the `[rn_dual_node_relu_bwd](https://github.com/rajiv256/NeuralCRNGen/blob/nonlinear_regression_z2_dotprod/julia/reactions_dotprod_simpler.jl#L64)` CRN.  
