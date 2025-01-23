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
using Distributions;


# Simulate a custom ODE
function simulate_reaction_network(network, u0, p;tspan=(), rate=1.0, reltol=1e-8, abstol=1e-8, kwargs...)
    # Network parameter variables
    oprob = ODEProblem(network, u0, tspan, p)
    sol = solve(oprob, Tsit5(), reltol=1e-8, abstol=1e-8, kwargs...)
    return sol
end


function custom_struct_round(a; precision_digits=10)
    for i in eachindex(a)
        a[i] = round(a[i], digits=precision_digits)
    end
    return a
end


function train_val_split(dataset; split=0.8)
    # An array of of data and class with class being the scalar in the final index
    d = Dict()
    Random.shuffle!(dataset)
    for item in dataset
        try
            push!(d[item[end]], item)
        catch 
            d[item[end]] = [item]
        end
    end
    train = Dict()
    val = Dict()
    for k in keys(d)
        train[k] = d[k][1:convert(Int32, split*length(d[k]))]
        val[k] = d[k][convert(Int32, split*length(d[k]))+1:end]        
    end
    train_dataset = []
    val_dataset = []
    for k in keys(d) # classes
        append!(train_dataset, train[k])
        append!(val_dataset, val[k])
    end
    Random.shuffle!(train_dataset)
    Random.shuffle!(val_dataset)
    return train_dataset, val_dataset
end


# Create a random matrix
function create_random_2D(r, c)
    rng = MersenneTwister()
    mat = randn(rng, r, c)
    return mat
end


# Get a random element from the dataset
function get_one(dataset, index=1)
    data = dataset[index]
    x = data[1:end-1]
    y = data[end]
    return x, y
end


function generate_annihilation_reactions(vars)
    
    for k in keys(vars)
        if startswith(k, "A") || startswith(k, "G") || startswith(k, "M") || startswith(k, "E")
            println("1.0, ", k, " --> 0")
        end
        if startswith(k, "W") && endswith(k, "p")
            sp = k
            dualsp = replace(k, "p"=>"m")
            println("1.0, ", sp, " + ", dualsp, " --> 0")
        end
    end
end


function get_index_of(prefix, vec)
    ret = []
    for i in eachindex(vec)
        vec_i = vec[i]
        if startswith("$vec_i", prefix)
            push!(ret, i)
        end
    end
    
    @assert length(ret) == 1 "Prefix: $prefix, vec: $vec"
    return ret[1]
end


function get_species_array(rn)
    ret = []
    for s in species(rn)
        push!(ret, replace(string(s), "(t)" => ""))
    end
    return ret
end


function create_node_params(dims; t0=0.0, t1=1.0, precision=10)
    theta = rand(Normal(0, 1),dims^2)
    for i in eachindex(theta)
        theta[i] = round(theta[i], digits=precision)
    end
    
    params = []
    append!(params, theta)
    push!(params, t0)
    push!(params, t1)
    # w = randn(dims)
    w = ones(dims)
    for i in eachindex(w)
        w[i] = round(w[i], digits=precision)
    end
    append!(params, w)
    return params
end

function create_node_params_reduced(dims; t0=0.0, t1=1.0, precision=10, h=0.2)
    theta = rand(Normal(0.2, 0.5), dims)
    theta = abs.(theta)

    params = []
    append!(params, theta)
    push!(params, t0)
    push!(params, t1)
    # w = randn(dims)
    w = ones(dims)
    
    append!(params, w)
    push!(params, h)

    return params
end


# Adds `k` zeroes to the end of the column matrix `u`
function augment(x, k=1) # Verified!
    ret = copy(x)
    for i in 1:k
        ret = vcat(ret, 0.0)
    end
    return ret
end


function sequester_params(p, dims)
    theta = zeros(dims, dims)
    for i in 1:dims^2
        theta[(i-1)÷dims + 1, (i-1)%dims + 1] = p[i]
    end
    t0 = p[dims^2 + 1]
    t1 = p[dims^2 + 2]
    w = p[dims^2+3:end]
    return theta, t0, t1, w
end

function sequester_params_reduced(p, dims)
    theta = zeros(dims, dims)
    for i in 1:dims
        theta[i] = p[i]
    end
    t0 = p[dims+1]
    t1 = p[dims+2]
    w = p[dims+3:dims+2 + dims]
    h = p[dims+2+dims+1]
    return theta, t0, t1, w
end