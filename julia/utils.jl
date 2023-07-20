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


# Simulate a custom ODE
function simulate_reaction_network(network, u0, p;tspan=(0.0, 1.0), rate=1.0, reltol=1e-8, abstol=1e-8, kwargs...)
    # Network parameter variables
    oprob = ODEProblem(network, u0, tspan, p)
    sol = solve(oprob, Rodas4(), reltol=reltol, abstol=abstol, kwargs...)
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

