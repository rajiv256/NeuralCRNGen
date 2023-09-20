using Pkg;
# Pkg.add("ReactionNetworkImporters")
# Pkg.add("Dictionaries")
# Pkg.add("LaTeXStrings")
# Pkg.add("Statistics")
# Pkg.add("ColorSchemes")
# Pkg.add("IterTools"); 
# Pkg.add("NNlib")

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
using NNlib;

include("datasets.jl")
include("utils.jl")


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
function forward_ffnet(z, w; threshold=nothing)
    yhat = dot(w, z) # Verified!
    # CHECK: Thinking of the final layer as a binary perceptron 
    println("ODE | yhat at t=T: $yhat")
    
    return yhat
end


function forward_step(u0, theta, w, tspan; threshold=nothing)
    # Output from the neural ode
    node_out = forward_node(u0, theta, tspan)
    # Extracting hidden state
    z = node_out.u[end][1:length(u0)]
    
    yhat = forward_ffnet(z, w, threshold=threshold)
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


function training_step(x, y, p; threshold=nothing)
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
    z, yhat = forward_step(x, theta, w, tspan, threshold=threshold)

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
    
    # Note that gradients[end] already contains gradient for t0
    append!(gradients, dldt1) # gradient for t1
    
    # Gradients wrt w
    wgrads = (yhat-y)*z
    println("ODE | error: ", yhat-y)
    println("ODE | wgrads, z: ", wgrads, z)
    for i in eachindex(wgrads)
        push!(gradients, wgrads[i])
    end
    println("ODE | Final layer gradients | ", wgrads)
    println("ODE | Gradients at t=0 | ", gradients)
    return z, yhat, loss, gradients
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
    println("==============ODE END=============")
    return params
    
end


# ################################################################# 

function node_main(params, train, val; dims=2, EPOCHS=20, LR=0.001, threshold=nothing)
    # Begin the training process
    losses = []
    val_losses = []
    for epoch in 1:EPOCHS
        epoch_loss = 0.0
        sleep(2)
        for i in eachindex(train)
            println("=========EPOCH: $epoch | ITERATION: $i ===========")
            x, y = get_one(train, i)

            # Augment
            x = augment(x, dims-length(x))
            println("ODE | Input: $x | Target: $y")
            println("params before | ", params)
            z, yhat, loss, gradients = training_step(x, y, params, threshold=threshold)
            epoch_loss += loss
            
            for param_index in eachindex(gradients)
                params[param_index] -= LR * gradients[param_index]
            end
            params[dims^2+1] = 0.0
            params[dims^2+2] = 1.0
            println("params at t=0 after update | ", params)
        end
        epoch_loss /= length(train)
        push!(losses, epoch_loss)
        lossplts = plot(losses)
        png(lossplts, "trainlossplts.png")
        accuracy = 0.0
        val_epoch_loss = 0.0
        before = []
        after = []
        yhats = []
        for v in eachindex(val)
            println("=======VAL Epoch: $epoch | ITERATION: $v")
            x, y = get_one(val, v)

            # Augment
            x = augment(x, dims - length(x))
            dims = length(x)

            theta, t0, t1, w = sequester_params(params, dims)
            tspan = (t0, t1)
            @assert length(w) == dims

            # Forward & Hidden state calculation
            println("ODE | w at t=0 | ", w)

            before_tmp = []
            append!(before_tmp, x)
            push!(before_tmp, y)
            push!(before, before_tmp)

            println("ODE | Input: $x | Target: $y")
            println("params before | ", params)
            z, yhat = forward_step(x, theta, w, tspan, threshold=threshold)
            loss = 0.5 * (yhat - y)^2
        
            class = math.ceil(yhat)
            after_tmp = []
            append!(after_tmp, z)
            push!(after_tmp, y)
            push!(after, after_tmp)


            val_epoch_loss += loss
            println("params | ", params)

            yhats_tmp = []
            append!(yhats_tmp, x)
            push!(yhats_tmp, class)
            push!(yhats, yhats_tmp)

            if class == y
                accuracy += 1
            end
        end
        if dims == 2
            beforeplt = scatter(getindex.(before, 1), getindex.(before, 2), group=getindex.(before, 3))
            afterplot = scatter(getindex.(after, 1), getindex.(after, 2), group=getindex.(after, 3))
            yhatplt = scatter(getindex.(yhats, 1), getindex.(yhats, 2), group=getindex.(yhats, 3))
        end
        if dims==3
            beforeplt = scatter3d(getindex.(before, 1), getindex.(before, 2), getindex.(before, 3), group=getindex.(before, 4))
            afterplot = scatter3d(getindex.(after, 1), getindex.(after, 2), getindex.(after, 3), group=getindex.(after, 4))
            yhatplt = scatter3d(getindex.(yhats, 1), getindex.(yhats, 2), getindex.(yhats, 3), group=getindex.(yhats, 4))
        end
        png(beforeplt, "before.png")
        png(afterplot, "after.png")
        println("accuracy: ", accuracy / length(val))
        
        
        png(yhatplt, "yhats.png")

    end
    
end

function neuralode(; DIMS=3)
    # train = create_linearly_separable_dataset(100, linear, threshold=0.0)
    # val = create_linearly_separable_dataset(40, linear, threshold=0.0)
    train = create_annular_rings_dataset(150)
    val = create_annular_rings_dataset(50)  
    # val = train   

    params_orig = create_node_params(DIMS, t0=0.0, t1=1.0)
    node_main(params_orig, train, val, dims=DIMS, EPOCHS=30, threshold=0.0, LR=0.001)
end

# neuralode()

