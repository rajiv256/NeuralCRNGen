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

function f(z, x, p, t)
    dims, theta, beta, w, h, _, _ = sequester_params(p)
    hvec = ones(dims)*h 
    fmat = (theta*x + beta).*z - z.*z + hvec
    @assert length(fmat) == dims
    return fmat
end

function forward!(du, u, xAndp, t)
    """
    xAndp: [x, dims, theta, beta, w, h, t0, t1]
    """
    dims = length(u)
    x = xAndp[1:dims]
    p = xAndp[dims+1:end]
    # _, theta, beta, w, h, _, _ = sequester_params(p)
    func = f(u, x, p, t)
    
    for i in eachindex(func)
        du[i] = func[i]
    end
end


# Calculates the final hidden state of the neural ode
function forward_node(u0, xAndp, tspan)
    prob = ODEProblem(forward!, u0, tspan, xAndp)
    sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-6, save_on=false)
    return sol
end


function backward!(du, u, sAndp, t)
    slen = length(u)
    s0 = sAndp[1:slen]
    p = sAndp[slen+1:end]
    func = f(u, s0, p, t)

    for i in eachindex(func)
        du[i] = func[i]
    end
end


function backward_node(s0, sAndp, tspan)
    prob = ODEProblem(backward!, s0, tspan, sAndp)
    sol = solve(prob, Tsit5(), reltol=1e-3, abstol=1e-6, save_on=false)
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