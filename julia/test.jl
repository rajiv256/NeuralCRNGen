using Pkg;
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

x = [1, 0]
p =  [1, 1, 1, 1, 1]
xAndp = (x, p)
a, b = xAndp
println(a)
println(b)