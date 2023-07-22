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

include("utils.jl")

function linear(x1, x2)
    return x1 + x2
end

function bilinear(x1, x2)
    return x1*x2
end

function mytanh(x1, x2)
    return tanh(2*x1 + 3*x2)
end

CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
file = open("/Users/rajiv/Desktop/PhD/RIP/neural-ode/data/iris/iris.data", "r")
lines = readlines(file)

function create_iris_dataset(feature_indices=[1, 3], classes=["Iris-setosa", "Iris-virginica"], N=100)
    dataset = []
    inputs = Vector{Vector{Float64}}()
    targets = Vector{String}()
    for line in lines
        row = split(line, ",")
        if row[end] in classes
            feature = Vector{Float64}()
            for fi in feature_indices
                push!(feature, parse(Float64, convert(String, row[fi])))
            end
            append!(dataset, [feature, row[end]]) 
            push!(inputs, feature)
            push!(targets, row[end])
        end
    end

    iris_dataset = []
    for (input, target) in zip(inputs, targets)
        data_item = Vector{Float64}()
        append!(data_item, input)
        if target == "Iris-setosa"
            push!(data_item, 0.0)
        end
        if target == "Iris-virginica"
            push!(data_item, 1.0)
        end
        data_item = reshape(data_item, (length(data_item), 1))
        push!(iris_dataset, data_item)
    end
    Random.shuffle!(iris_dataset)
    # To be used for training

    iris_train_dataset, iris_val_dataset = train_val_split(iris_dataset)

    return iris_train_dataset, iris_val_dataset, iris_val_dataset
end


function plot_iris_dataset(dataset=[])
    plot()
    for data in dataset
        scatter!(data)
    end 
end


rng = MersenneTwister(256)

# This creates a regression dataset.
function create_dataset!(n, yfunc)
    dataset = []
    for i in 1:n
        x1 = abs(convert(Float64, randn(rng, 1)[1]))
        x2 = abs(convert(Float64, randn(rng, 1)[1]))
        y = yfunc(x1, x2) # Only positive for now.
        y = convert(Float64, y)
        data_item = Vector{Float64}()
        append!(data_item, [abs(x1), abs(x2), y])
        data_item = reshape(data_item, (length(data_item), 1))
        push!(dataset, data_item)
    end
    return dataset;
end

function create_linearly_separable_dataset(n, yfunc; threshold=1.0)
    dataset = []
    nneg = 0
    for i in 1:n
        x1 = convert(Float64, randn(rng, 1)[1])
        x2 = convert(Float64, randn(rng, 1)[1])
        y = convert(Float64, yfunc(x1, x2)) # only positive for now
        if y > threshold
            y = 1.0
        else
            y = 0.0
            nneg += 1
        end
        data_item = Vector{Float64}()
        append!(data_item, [x1, x2, y])
        data_item = reshape(data_item, (length(data_item), 1))
        push!(dataset, data_item)
    end
    print("nneg: ", nneg, " npos: ", length(dataset)-nneg)
    return dataset;
end


# This is a classification dataset with nonlinearly separable data
function create_annular_rings_dataset(n, r)
    dataset = []
    nneg = 0
    for i in 1:n
        x1 = convert(Float64, randn(rng, 1)[1]) # CHECK
        x2 = convert(Float64, randn(rng, 1)[1]) # CHECK
        y = 1
        if norm([x1, x2]) < r
            y = 0
            nneg += 1
        end
        # Leaving some space between annular rings
        if norm([x1, x2]) > r && norm([x1, x2]) < 1.5*r
            continue
        end
        
        data_item = Vector{Float64}()
        append!(data_item, [abs(x1), abs(x2), y])
        data_item = reshape(data_item, (length(data_item), 1))
        push!(dataset, data_item)
    end
    print("nneg: ", nneg)
    Random.shuffle!(dataset)
    return dataset
end

# # This part has to be changed into a function call 
# N = 100
# Nval = 20
# rings_train_dataset = create_annular_rings_dataset!(N, 1.0)
# rings_val_dataset = create_annular_rings_dataset!(Nval, 1.0)

# tanh_train_dataset = create_dataset!(N, mytanh)
# tanh_val_dataset = create_dataset!(Nval, mytanh)
# print(tanh_train_dataset[1])

# linear_class_train_dataset = create_linearly_separable_dataset!(N, linear, threshold=0.0)
# linear_class_val_dataset = create_linearly_separable_dataset!(Nval, linear, threshold=0.0)


# Create a random matrix
function create_random_2D(r, c)
    rng = MersenneTwister()
    mat = randn(rng, r, c)
    return mat
end