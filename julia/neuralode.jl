using Pkg;
# Pkg.add("ReactionNetworkImporters")
# Pkg.add("Dictionaries")
# Pkg.add("LaTeXStrings")
# Pkg.add("Statistics")
# Pkg.add("ColorSchemes")
# Pkg.add("IterTools"); 

using DifferentialEquations;
using Random;
using Plots;
using Formatting;
using LinearAlgebra;
using Noise;
using ReactionNetworkImporters;
using Dictionaries;
using LaTeXStrings;
using Statistics;
using ColorSchemes;
using Catalyst;
using IterTools;

include("datasets.jl")
include("utils.jl")
include("reactions2D.jl")


function f(z, theta)
    """
    Args:
        u: hidden state
        theta: resnet parameters of size [length(u), length(u)]
    """
    fmat = theta * z
    @assert length(fmat) == length(z)
    return fmat
end


function forward!(du, u, theta, t)
    func = f(u, theta)
    for i in eachindex(func)
        du[i] = func[i]
    end
end


# Calculates the final hidden state of the neural ode
function forward_node(u0, theta, tspan)

    prob = ODEProblem(forward!, u0, tspan, theta)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end


# Final feedforward layer similar to a perceptron
function forward_ffnet(z, w)
    yhat = dot(w, z) # Verified!
    return yhat
end


function forward_step(u0, theta, w, tspan)
    # Output from the neural ode
    node_out = forward_node(u0, theta, tspan)
    # Extracting hidden state
    z = node_out.u[end][1:length(u0)]
    yhat = forward_ffnet(z, w)
    return (z, yhat)
end


function aug_dynamics!(du, u, theta, t)
    dims = Int32(length(theta) ÷ sqrt(length(theta)))

    # Time dynamics for the hidden state
    offset = 0
    z = u[1:dims]
    func = f(z, theta)

    @assert length(func) == dims

    for i in 1:dims
        du[offset+i] = func[i]
    end

    offset += dims


    # Time dynamics for the adjoint
    a = u[dims+1:2*dims]
    a = reshape(a, (dims, 1))

    # ∂f/∂z = 𝜃
    dfdz = theta
    @assert size(theta) == (dims, dims)

    dadt = reshape(-transpose(a) * dfdz, dims)
    for i in eachindex(dadt)
        du[offset+i] = dadt[i]
    end

    offset += length(dadt)
    @assert offset == 2 * dims # offset after adding dzdt and dady


    # Time dynamics for gradients
    dfdtheta = zeros(dims, dims^2)

    for i in 1:dims
        for j in 1:dims
            dfdtheta[i, (i-1)*dims+j] = z[j]
        end
    end

    dgrads = -transpose(a) * dfdtheta

    @assert size(dgrads) == (1, dims^2)

    for i in eachindex(dgrads)
        du[offset+i] = dgrads[i]
    end
    offset += length(dgrads)

    # Time dynamics of time(!!): Not used though sigh.
    # TODO: Might wanna change this in future if things don't work
    dfdt = zeros(dims, 1)
    tgrads = -transpose(a) * dfdt

    @assert length(tgrads) == 1

    # currently not changing time!
    du[offset+1] = 0

end


function backpropagation_step(s0, theta, tspan)
    prob = ODEProblem(aug_dynamics!, s0, tspan, theta)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end


function training_step(x, y, p)
    """
    Args:
        x: augmented input
        y: output 
        p: parameters of the entire network
    """
    dims = length(x)
    
    theta, t0, t1, w = sequester_params(p, dims)
    tspan = (t0, t1)
    @assert length(w) == dims 
    
    # Forward & Hidden state calculation
    println("ODE | w at t=0 | ", w)
    z, yhat = forward_step(x, theta, w, tspan)
    z = reshape(z, (dims, 1)) # Make z a row-vector
    println("ODE | z at t=T | ", z)
    # Loss
    loss = 0.5*(yhat-y)^2
    
    # Adjoint calculation
    a = (yhat-y)*w
    a = reshape(a, (dims, 1))
    println("ODE | yhat at t=T | ", yhat)
    println("ODE | Adjoint at t=T | ", a)
    
    # Initial theta gradients
    gtheta = zeros(dims^2, 1)
    println("ODE | Gradients at t=T | ", gtheta)
    
    # Initial time gradients 
    func = f(z, theta)

    dldt1 = -transpose(a)*f(z, theta)
    dldt1 = convert(Array{Float64}, dldt1)
    
    # Initial state for the reverse time ODE
    s0 = vcat(z, a, gtheta, dldt1)
    rtspan = reverse(tspan)
    
    backward = backpropagation_step(s0, theta, rtspan)
    println("ODE | Adjoint at t=0 | ", backward.u[end][dims+1:2*dims])
    
    gradients = backward.u[end][2*dims+1:end]
    gradients = reshape(gradients, size(gradients)[1])
    println("ODE | Gradients at t=0 |  ", backward.u[end][2*dims+1:2*dims + dims^2])
    
    # Note that gradients[end] already contains gradient for t0
    append!(gradients, dldt1) # gradient for t1
    
    # Gradients wrt w
    wgrads = (yhat-y)*z
    wgrads = wgrads * 0  # rajiv: freezing final layer, 2 jun 2024
    println("ODE | error: ", yhat-y)
    println("ODE | wgrads, z: ", wgrads, z)
    for i in eachindex(wgrads)
        push!(gradients, wgrads[i])
    end
    println("ODE | Final layer gradients | ", wgrads)
    return z, yhat, loss, gradients
end


function predict(x, p)
    theta, t0, t1, w = sequester_params(p, length(x))
    tspan = (t0, t1)
    z, yhat = forward_step(x, theta, w, tspan) #TODO: Check tspan
    return yhat
end

function one_step_node(x, y, params, LR, dims)
    println("=======ODE==================")
    println("ODE | Input: $x | Target: $y")
    println("params before | ", params)
    z, yhat, loss, gradients = training_step(x, y, params)

    # Parameter update
    println("ODE | gradients | ", gradients)
    for param_index in eachindex(gradients)
        params[param_index] -= LR * gradients[param_index]
    end
    params[dims^2+1] = 0.0
    params[dims^2+2] = 1.0
    println("params | ", params)
    return params
    println("==============ODE END=============")
end


# ################################################################# 

function node_main(params, train, val; dims=2, EPOCHS=10, LR=0.001)
    # Begin the training process
    losses = []
    val_losses = []
    for epoch in 1:EPOCHS
        epoch_loss = 0.0
        for i in eachindex(train)
            x, y = get_one(train, i)

            # Augment
            x = augment(x, dims-length(x))
            println("ODE | Input: $x | Target: $y")
            println("params before | ", params)
            z, yhat, loss, gradients = training_step(x, y, params)
            epoch_loss += loss

            # Parameter update
            println("ODE | gradients | ", gradients)
            for param_index in eachindex(gradients)
                params[param_index] -= LR * gradients[param_index]
            end
            params[dims^2+1] = 0.0
            params[dims^2+2] = 1.0
            println("params | ", params)

        end
        epoch_loss /= length(train)
        push!(losses, epoch_loss)
        tr_png = plot(range(1, epoch), losses)
        png(tr_png, "training_losses.png")

        val_epoch_loss = 0.0
        for i in eachindex(val)
            x, y = get_one(val, i)

            # Augment
            x = augment(x, dims - length(x))

            println("ODE | Input: $x | Target: $y")
            println("params before | ", params)
            z, yhat, loss, gradients = training_step(x, y, params)
            val_epoch_loss += loss
            println("params | ", params)

        end
        val_epoch_loss /= length(val)
        push!(val_losses, val_epoch_loss)
        val_png = plot(range(1, epoch), val_losses)
        png(val_png, "validation_losses.png")
    end
end


function neuralode(; DIMS=2)
    train = create_linearly_separable_dataset(100, linear, threshold=0.0)
    val = create_linearly_separable_dataset(40, linear, threshold=0.0)
    params_orig = create_node_params(DIMS, t0=0.0, t1=1.0)
    node_main(params_orig, train, val)
end

# neuralode()

