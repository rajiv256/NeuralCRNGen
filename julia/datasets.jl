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
    return x1 * x2
end

function mytanh(x1, x2)
    return tanh(2 * x1 + 3 * x2)
end

CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
file = open("data/iris/iris.data", "r")
lines = readlines(file)

"""
 Number of Instances: 150 (50 in each of three classes)
6. Number of Attributes: 4 numeric, predictive attributes and the class
7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
"""

function create_iris_dataset(;feature_indices=[1, 3], classes=["Iris-setosa", "Iris-virginica"], N=100)
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
        println(target)
        if target == "Iris-versicolor"
            push!(data_item, 0.0)
        end
        if target == "Iris-virginica"
            push!(data_item, 1.0)
        end
        # data_item = reshape(data_item, (length(data_item), 1))
        push!(iris_dataset, data_item)
    end
    Random.shuffle!(iris_dataset)
    # To be used for training

    # iris_train_dataset, iris_val_dataset = train_val_split(iris_dataset)
    iris_train_dataset = iris_dataset[1:Int32(0.8*length(iris_dataset))]
    iris_val_dataset = iris_dataset[Int32(0.8 * length(iris_dataset))+1:end]

    return iris_train_dataset, iris_val_dataset
end


function plot_iris_dataset(dataset=[])
    plot()
    gg = nothing 
    gg = scatter!(getindex.(dataset, 1), getindex.(dataset, 2), group=getindex.(dataset, 3))
    return gg
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
    return dataset
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
    print("nneg: ", nneg, " npos: ", length(dataset) - nneg)
    return dataset
end


# This is a classification dataset with nonlinearly separable data
function create_annular_rings_dataset(n; lub=0.04, lb=0.3, mb=0.8, ub=1.0)
    
    dataset = []
    center = 0
    normal = Uniform(-1, 1)
    while length(dataset) <= n÷2
        x1 = convert(Float32, rand(normal, 1)[1])
        x2 = convert(Float32, rand(normal, 1)[1])
        if norm([x1-center x2-center]) <= lb && norm([x1-center x2-center]) >= lub
            y = 0.0
            push!(dataset, [x1 x2 y])
        end
    end

    while length(dataset) <= n
        x1 = convert(Float32, rand(normal, 1)[1])
        x2 = convert(Float32, rand(normal, 1)[1])
        if norm([x1-center x2-center]) >= mb && norm([x1-center x2-center]) <= ub
            y = 1.0
            push!(dataset, [x1 x2 y])
        end
    end
    Random.shuffle!(dataset)
    return dataset
end

function create_xor_dataset(n; pos=1.0, neg=0.0, threshold=0.5)
    # xor_dataset = create_xor_dataset(100)

    # g = scatter!(getindex.(xor_dataset, 1), getindex.(xor_dataset, 2), group=getindex.(xor_dataset, 3))
    # png(g, "julia/images/xor_dataset.png")
    dataset = []
    uniform = Uniform(0, 1)

    while length(dataset) <= n÷2
        x1 = convert(Float32, rand(uniform, 1)[1])
        x2 = convert(Float32, rand(uniform, 1)[1])
        
        x1binary = Bool(floor(x1 + 0.5))
        x2binary = Bool(floor(x2 + 0.5))

        y = Float32(x1binary ⊻ x2binary)
        if y == 0.0
            push!(dataset, [x1 x2 neg])    
        end
    end

    while length(dataset) <= n

        x1 = convert(Float32, rand(uniform, 1)[1])
        x2 = convert(Float32, rand(uniform, 1)[1])
        
        x1binary = Bool(floor(x1 + 0.5))
        x2binary = Bool(floor(x2 + 0.5))

        y = Float32(x1binary ⊻ x2binary)
        if y == 1.0
            push!(dataset, [x1 x2 pos])
        end
    end
    Random.shuffle!(dataset)
    return dataset
end

function create_and_dataset(n; pos=1.0, neg=0.0, threshold=0.5)
    
    dataset = []
    uniform = Uniform(0, 1)

    while length(dataset) <= n÷2
        x1 = convert(Float32, rand(uniform, 1)[1])
        x2 = convert(Float32, rand(uniform, 1)[1])
        
        x1binary = Bool(floor(x1 + 0.5))
        x2binary = Bool(floor(x2 + 0.5))

        y = Float32(x1binary & x2binary)
        if y == 0.0
            push!(dataset, [x1 x2 neg])
        end
    end

    while length(dataset) <= n

        x1 = convert(Float32, rand(uniform, 1)[1])
        x2 = convert(Float32, rand(uniform, 1)[1])

        x1binary = Bool(floor(x1 + 0.5))
        x2binary = Bool(floor(x2 + 0.5))

        y = Float32(x1binary & x2binary)
        if y == 1.0
            push!(dataset, [x1 x2 pos])
        end
    end
    Random.shuffle!(dataset)
    return dataset
end
# # # This part has to be changed into a function call 
# N = 150
# # Nval = 20
# rings_train_dataset = create_annular_rings_dataset(N)
# # rings_val_dataset = create_annular_rings_dataset!(Nval, 1.0)

# # tanh_train_dataset = create_dataset!(N, mytanh)
# # tanh_val_dataset = create_dataset!(Nval, mytanh)
# # print(tanh_train_dataset[1])

# # linear_class_train_dataset = create_linearly_separable_dataset!(N, linear, threshold=0.0)
# # linear_class_val_dataset = create_linearly_separable_dataset!(Nval, linear, threshold=0.0)


# # Create a random matrix
# function create_random_2D(r, c)
#     rng = MersenneTwister()
#     mat = randn(rng, r, c)
#     return mat
# end

# plt = scatter(getindex.(rings_train_dataset, 1), getindex.(rings_train_dataset, 2), group=getindex.(rings_train_dataset, 3))
# png(plt, "annular_rings.png")


train, val = create_iris_dataset(feature_indices=[1, 3])
# println(train)
trplt = plot_iris_dataset(train)
valplt = plot_iris_dataset(val)

png(trplt, "images/iris_train.png")
png(valplt, "images/iris_val.png")