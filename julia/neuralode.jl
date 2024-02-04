using Pkg;
# Pkg.add("ReactionNetworkImporters");
# Pkg.add("Dictionaries");
# Pkg.add("LaTeXStrings");
# Pkg.add("Statistics");
# Pkg.add("ColorSchemes");
# Pkg.add("IterTools"); 
# Pkg.add("NNlib"); 
# Pkg.add("DifferentialEquations");
# Pkg.add("Plots");
# Pkg.add("Formatting");
# Pkg.add("LinearAlgebra");
# Pkg.add("Noise");
# Pkg.add("Catalyst");

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


function f(u, xAndp, t)
    x = xAndp[1:3]
    p = xAndp[4:end]
    dims, theta, beta, w, h, _, _ = sequester_params(p)
    hvec = [h, h, h]
    fmat = hvec + (theta * x + beta) .* u - u .* u
    # fmat = theta*u
    @assert length(fmat) == length(u)
    return fmat
end


function forward!(du, u, xAndp, t)
    fmat = f(u, xAndp, t)
    for i in eachindex(fmat)
        du[i] = fmat[i]
    end
end


# Calculates the final hidden state of the neural ode
function forward_node(u0, xAndp, tspan)
    prob = ODEProblem(forward!, u0, tspan, xAndp)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-12, save_on=false)
    return sol
end


function node_main(params, train, val; DIMS=3, EPOCHS=20, LR=0.001, threshold=nothing)
    x, y = get_one(train, 1)
    # Augment
    x = augment(x, DIMS-length(x))
    # for j in 1:length(x)
    #     x[j] = abs(x[j])
    # end
    println("ODE | Input: $x | Target: $y")
    dims, theta, beta, w, h, t0, t1 = sequester_params(params)
    
    println("params before | ", params)
    xAndp = vcat(x, params)
    tspan = (t0, t1)
    sol = forward_node(x, xAndp, tspan)
    println("Ideal ReLU | ", relu.(theta*x + beta))
    println("ODE | z at t=T | ", sol[end])

    z = reshape(sol[end], (DIMS, 1)) 
    s0 = z
    sAndp = vcat(s0, params)
    rtspan = reverse(tspan)
    bsol = backward_node(s0, sAndp, rtspan)
    println("ODE | Input: $x ")
    println("ODE | z at t=0 | ", bsol[end])
end

function neuralode(; DIMS=3)
    # train = create_linearly_separable_dataset(100, linear, threshold=0.0)
    # val = create_linearly_separable_dataset(40, linear, threshold=0.0)
    train = create_annular_rings_dataset(150)
    val = create_annular_rings_dataset(50)
    # val = train   
    params_orig = create_node_params(DIMS, t0=0.0, t1=4.0)
    for i in eachindex(params_orig)
        params_orig[i] = abs(params_orig[i])
    end 
    node_main(params_orig, train[1:2], val[1:2], DIMS=DIMS, EPOCHS=30, threshold=0.0, LR=0.001)
end

neuralode()

# 