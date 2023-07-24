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

function f(x)
    push!(x, 4)
end

x = [1; 2; 3]
f(x)
println(x)
plot_augmented_state(copy(vars), train)
train_acc = calculate_accuracy(train, copy(vars), tspan=(0.0, 1.0))
_print_vars(vars, "Z", title="After calculate_accuracy train-- parameters.")
val_acc = calculate_accuracy(val, copy(vars), tspan=(0.0, 1.0))

@show epoch_loss, train_acc, val_acc
return

push!(crn_epoch_train_accs, train_acc)
push!(crn_epoch_val_accs, val_acc)

plt_train = plot(crn_epoch_train_accs)
plt_val = plot(crn_epoch_val_accs)
png(plt_train, "crn_epoch_train_accuracies.png")
png(plt_val, "crn_epoch_val_accuracies.png")