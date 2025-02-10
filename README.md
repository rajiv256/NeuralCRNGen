# Preliminaries 

Make sure you have `Julia v1.11.0` or above installed. 

# Execution

To execute, navigate to the root directory `NeuralCRNGen/` and execute

```
julia julia/neuralcrn_linear_reduced.jl
```

_Note: If you don't have certain packages installed, execution might throw an error. In that case, uncomment the `Pkg.add("DifferentialEquations")` (for example) at the top of the file._

This command executes the `neuralcrn()` function. One important parameter that needs to be provided is the `output_dir`. It takes the value of a  folder name and creates the folder at `julia/<output_dir>`. All the image files and the output files will be populated into this folder. 

# Useful Files and Directories

- The CRNs could be found in `julia/linear_reduced.jl`.
- The function for creating the dataset could be found in `julia/datasets.jl`.
- The util functions for creating the plots could be found in `julia/myplots.jl`.
- Other general utils functions can be found in `julia/utils.jl`.

# Important functions

- The code for training the Neural CRN model is in `crn_main(...)` function. 
