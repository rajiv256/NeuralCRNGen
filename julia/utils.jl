import Pkg; 
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
using Catalyst;

# Simulate a custom ODE
function simulate_reaction_network(network, u0, rate_constants;tspan=(), rate=1.0, kwargs...)
    # Network parameter variables
    oprob = ODEProblem(network, u0, tspan, rate_constants)
    sol = solve(oprob, TRBDF2(autodiff=false), reltol=1e-3, abstol=1e-8, maxiters=1e6)
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


function create_node_params(dims; t0=0.0, t1=1.0, h=0.5, precision=10, NUM_CLASSES=3)
    params = []

    push!(params, Float32(dims))

    theta = rand(Normal(0.0, 0.2), dims^2)
    theta = theta/sqrt(dims)

    append!(params, theta)
    beta = ones(dims)*0.1 
    append!(params, beta)

    w = rand(Normal(0.0, 0.2), dims*NUM_CLASSES)
    w = w/sqrt(dims)
    append!(params, w)

    push!(params, h)

    push!(params, t0)
    push!(params, t1)

    for i in eachindex(params)
        params[i] = abs(params[i])
    end
    
    return params
end

# Adds `k` zeroes to the end of the column matrix `u`
function augment(x, k; augval=1.0) # Verified!
    ret = copy(x)
    for i in 1:k
        ret = vcat(ret, augval)
    end
    return ret
end


function sequester_params(p; NUM_CLASSES=3) 
    offset = 1
    # offset + sz 
    dims = Int32(p[offset])
    offset += 1
    
    theta = p[offset:offset + dims^2 - 1]
    println(theta)
    theta = reshape(theta, (dims, dims))

    offset += dims^2
    
    beta = p[offset: offset + dims-1]
    offset += dims

    w = reshape(p[offset:offset + dims*NUM_CLASSES - 1], (dims, NUM_CLASSES))
    offset += dims*NUM_CLASSES

    h = p[offset]
    offset += 1

    t0 = p[offset]
    offset += 1

    t1 = p[offset]
    offset += 1
    @show dims
    @show theta
    @show beta 
    @show w
    @show h 
    @show t0
    @show t1
    return dims, theta, beta, w, h, t0, t1
end
