using Plots
using Random
using Distributions

# Define the function
f(x, y) = sin(x) + y^2

# Define the range for plotting
x_range = -0.5:0.001:3
y_range = -0.5:0.001:3
z_vals = [f(x, y) for x in x_range, y in y_range]

# Create surface plot
plot_surface = surface(x_range, y_range, z_vals, title="Surface Plot of z = sin(x) + y^2", xlabel="x", ylabel="y", zlabel="z", alpha=0.2)

# Generate training data
n_samples = 50
x_samples = rand(Uniform(0.5, 2), n_samples)
y_samples = rand(Uniform(0.5, 2), n_samples)
z_samples = f.(x_samples, y_samples)

# Scatter plot of training data
scatter!(x_samples, y_samples, z_samples, markersize=4, markercolor=:red, label="Training Data")

display(plot_surface)