# This file is used to run CRN operations
using Pkg;
Pkg.add("NNlib");
Pkg.add("ProgressMeter");

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
using ProgressMeter;
using Distributions;

include("datasets.jl")
include("utils.jl")
include("reactionsReLU.jl")
# include("reactionsReLUAlt.jl")
# include("neuralode.jl")
include("myplots.jl")

Random.seed!(42) 

theme(:default);
palette(:Dark2_5);

# Set global font sizes
default(
    legendfontsize=18,            # legend font size
    tickfontsize=16,              # tick font size
    guidefontsize=18,             # axis label font size
    titlefontsize=14,             # title font size
    # Optional: you can also set font family
    fontfamily="Arial",  # or "Helvetica", "Times New Roman", etc.,
    # Option 3: Using hex colors
    palette=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", 
           "#66CCEE", "#BBBBBB", "#EE3377", "#000000", "#CC6677",], 
    markersize=2,
    framestyle=:semi,
    # widen=false,
    lw=2, 
    size=(600, 450),  # Wider figure
    bottom_margin=3Plots.mm, # Add global margin
    legend=:best
)


function _convert_species2var(sp)
    ret = string(sp)
    ret = replace(ret, "(t)" => "")
    return ret
end


function _index2Dvar(sym, index, val; dims=3)
    second = (index-1)%dims + 1
    first = (index-1)÷dims + 1
    return Dict(
        "$(sym)$(first)$(second)p"=> max(0.0, val),
        "$(sym)$(first)$(second)m"=> max(0.0, - val)
    )
end


function _index1Dvar(sym, index, val; dims=3)  
    return Dict(
        "$sym$(index)p" => max(0.0, val),
        "$sym$(index)m" => max(0.0, -val)
    )
end


function _prepare_u(rn, vars)
    ss = species(rn)
    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)
    return u
end


function _print_vars(vars, prefix; title="")
    println(title, "---------------")
    Xvars = [k for k in keys(vars) if startswith(k, prefix)]
    sort!(Xvars)
    for xvp in Xvars
        if endswith(xvp, "p")
            xvm = replace(xvp, "p"=>"m")
            println("""$xvp: $(vars[xvp]) | $xvm: $(vars[xvm]) | $(vars[xvp]-vars[xvm])""")
        end
    end
    println("---------------------")
end


function _filter_rn_species(ss; prefix="Z")
    xs = filter(x->startswith(string(x), prefix), ss)
    return xs
end


function _form_vector(vars, prefix)
    names = []
    for name in keys(vars)
        if startswith(name, prefix)
            push!(names, name)
        end
    end
    ps = []
    ms = []
    for name in names
        if endswith(name, "p")
            push!(ps, name)
        elseif endswith(name, "m")
            push!(ms, name)
        end
    end
    sort!(ps)
    sort!(ms)
    ps = [vars[name] for name in ps]
    ms = [vars[name] for name in ms]
    ps = reshape(ps, (length(ps), 1))
    ms = reshape(ms, (length(ms), 1))
    mat = hcat(ps, ms)
    return mat
end


function _create_symbol_matrix(prefix, dims)
    @assert length(dims) == 2
    symbols = []
    rows = dims[1]
    cols = dims[2]
    for row in 1:rows
        if cols == 1
            push!(symbols, string(prefix, string(row), "p"))
            push!(symbols, string(prefix, string(row), "m"))
            continue
        end
        for col in 1:cols
            push!(symbols, string(prefix, string(row), string(col), "p"))
            push!(symbols, string(prefix, string(row), string(col), "m"))
        end
    end
    drdims = (dims..., 2)
    # symbols = reshape(symbols, drdims)
    symbols = dropdims(symbols, dims=tuple(findall(size(symbols) .== 1)...))
    return symbols
end


function _assign_vars(vars, sym_matrix, val_matrix)
    vm = collect(Iterators.flatten(val_matrix))
    for i in eachindex(sym_matrix)
        vars[string(sym_matrix[i])] = vm[i]
    end
end


function crn_error_binary_scalar_mult(vars, subS, subP; max_val=40.0)
    ss = species(rn_dual_binary_scalar_mult)

    # Convert species into the variables in vars
    varkeys = [_convert_species2var(sp) for sp in ss]
    varkeys = [replace(k, "S" => subS) for k in varkeys]
    varkeys = [replace(k, "P" => subP) for k in varkeys]

    u = [get!(vars, k, 0.0) for k in varkeys]
    p = []

    sol = simulate_reaction_network(rn_dual_binary_scalar_mult, u, p, tspan=(0.0, max_val))

    # Update the value only if it is a product
    for i in eachindex(ss)
        sp = string(ss[i])

        if startswith(sp, "P")  # It's a product species
            varkey = varkeys[i]  # original variable name
            vars[varkey] = sol[end][i]
        end
    end
    _print_vars(vars, "$subP", title="CRN | $subP at t=T")
end


function crn_output_annihilation(vars; max_val=40.0)
    ss = species(rn_output_annihilation)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []
    sol = simulate_reaction_network(rn_output_annihilation, u, p, tspan=(0.0, max_val))

    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
    _print_vars(vars, "O", title="CRN | Output after annihilation at t=T")
end


function crn_create_error_species(vars)
    # CHECK: Assuming that Y and O species are up to date
    ss = species(rn_create_error_species)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []

    # Due to the way in which `rn_create_error_species` is setup, the tspan has to be (0.0, 1.0)
    sol = simulate_reaction_network(rn_create_error_species, u, p, tspan=(0.0,1.0))
    println("CRN | error species | ", sol[end])
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
    _print_vars(vars, "E", title="CRN | Error species at t=T")
end


function plot_augmented_state(varscopy, dataset; tspan=(0.0, 1.0), dims=3, threshold=0.0, augval=augval, output_dir="")
    aug_x = []
    reg_x = []
    yhats = []
    markers = []
    circles = []

    for i in eachindex(dataset)
        x, y = get_one(dataset, i)

        x = augment(x, dims - length(x), augval=augval)
  
        for zi in eachindex(x)
            d = _index1Dvar("Z", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end
        for zi in eachindex(x)
            d = _index1Dvar("X", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end
        varscopy["Yp"] = y
        varscopy["Ym"] = 0.0
        elem = []
        append!(elem, x)
        push!(elem, y)
        push!(reg_x, elem)
        
        crn_dual_node_fwd(rn_dual_node_relu_fwd, varscopy, tspan=tspan)
    
        yhat = crn_dot(varscopy, "Z", "W", max_val=40.0)
        @show yhat, yhat[1] - yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
        output = 0.0
        if varscopy["Op"]-varscopy["Om"] > threshold
            output = 1.0
        end
        push!(
            aug_x, [
                varscopy["Z1p"] - varscopy["Z1m"],
                varscopy["Z2p"] - varscopy["Z2m"],
                varscopy["Z3p"] - varscopy["Z3m"],
                output
            ]
        )
        
        if y == 0.0
            push!(markers, :circ)
        else
            push!(markers, :rect)
        end
        push!(circles, :circ)

        temp = []
        append!(temp, x)
        push!(temp, output)
        push!(yhats, temp)

        
    end
    plot()
    myscatter3d(getindex.(reg_x, 1), getindex.(reg_x, 2), getindex.(reg_x, 3), getindex.(reg_x, 4), circles, output_dir=output_dir, name="crn_before_aug")
    plot()
    myscatter3d(getindex.(aug_x, 1), getindex.(aug_x, 2), getindex.(aug_x, 3), getindex.(aug_x, 4), markers, output_dir=output_dir, name="crn_after_aug")
    plot()
    myscatter(getindex.(yhats, 1), getindex.(yhats, 2), getindex.(yhats, 4), output_dir=output_dir, name="crn_yhats")
end


function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=3, threshold=0.0, markers=[:circle, :rect], augval=1.0, output_dir="")
    acc = 0
    preds2d = []
    wrongs = []

    for i in 1:length(dataset)
        x, y = get_one(dataset, i)
        
        temp = []
        append!(temp, x)

        x = augment(x, dims - length(x), augval=augval)

        for zi in eachindex(x)
            d = _index1Dvar("Z", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end
        for zi in eachindex(x)
            d = _index1Dvar("X", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end

        
        crn_dual_node_fwd(rn_dual_node_relu_fwd, varscopy, tspan=tspan)
        varscopy["Yp"] = y
        varscopy["Ym"] = 0.0

        # Calculate yhat            
        yhat = crn_dot(varscopy, "Z", "W", max_val=40.0)
        @show yhat, yhat[1]-yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
        output = 0.0
        if varscopy["Op"] - varscopy["Om"] >= threshold # TODO: CHECK BEFORE
            output = 1.0
        end
        
        # Casting a float into an integer
        push!(temp, output)
        push!(temp, y)  # temp: x1 x2 output y

        push!(preds2d, temp)
        if output == y
            acc += 1
        end

        if output != y
            push!(wrongs, x)
        end
    end

    plot()
    myscatter(getindex.(preds2d, 1), getindex.(preds2d, 2), getindex.(preds2d, 3), output_dir=output_dir, name="outputs", xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}")
    plot()
    gg = myscatter(getindex.(preds2d, 1), getindex.(preds2d, 2), 
                 getindex.(preds2d, 3), output_dir=output_dir, name="outputs")
    gg = myscatternogroup(getindex.(wrongs, 1), getindex.(wrongs, 2), markershape=:xcross, markersize=5, label="errors",
        output_dir=output_dir, name="outputs_with_wrongs", xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}")
    
    savefig(gg, "julia/$output_dir/images/outputs_with_wrongs.svg")
    savefig(gg, "julia/$output_dir/images/outputs_with_wrongs.png")
    
    println("Accuracy: $(acc/length(dataset))")
    return acc/length(dataset)
end


function dissipate_and_annihilate(vars, tspan)
    ss = species(rn_dissipate_reactions)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []

    sol = simulate_reaction_network(rn_dissipate_reactions, u, p, tspan=tspan)
    
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end

end


function crn_param_update(rn, vars, eta, tspan)

    ss = species(rn)

    u = [vars[_convert_species2var(sp)] for sp in ss]
    
    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1 k2]

    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    for i in eachindex(ss)
        if startswith(string(ss[i]), "P")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    for i in eachindex(ss)
        if startswith(string(ss[i]), "B")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
end


function crn_final_layer_update(vars, eta, tspan)
    ss = species(rn_final_layer_update)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1, k2]
    sol = simulate_reaction_network(rn_final_layer_update, u, p, tspan=tspan)

    for i in eachindex(ss)
        if startswith(string(ss[i]), "W")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
end


function crn_dual_backprop(rn, vars, tspan; bias=0.01, reltol=1e-4, abstol=1e-6, D=2, default=0.0)
    ss = species(rn)

    u = [vars[_convert_species2var(sp)] for sp in ss]
    @assert length(u) == length(ss)
    p = []
    sol = simulate_reaction_network(rn, u, p, tspan=tspan) # TODO: CHECK
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
end


function crn_subtract(a, b; max_val=40.0, default=0.0)
    u = [
        :Ap => a[1], :Am => a[2], :Bp => b[1], :Bm => b[2],
        :Yp => default, :Ym => default
    ]
    p = []
    sol = simulate_reaction_network(rn_dual_subtract, u, p, tspan=(0.0, max_val))

    ss = species(rn_dual_subtract)
    yp = sol[end][get_index_of("Yp", ss)]
    ym = sol[end][get_index_of("Ym", ss)]
    y = [yp ym]
    return y
end


function crn_mult(a, b; max_val=100.0)
    # a: [ap, am]   b: [bp, bm]

    u = [:Ap => a[1], :Am => a[2], :Bp => b[1], 
        :Bm => b[2], :Yp => 0, :Ym => 0]
    p = []
    sol = simulate_reaction_network(rn_dual_mult, u, p, tspan=(0.0, max_val))

    # Calculate value
    ss = species(rn_dual_mult)

    yp = sol[end][get_index_of("Yp", ss)]
    ym = sol[end][get_index_of("Ym", ss)]
    y = [yp ym]
    return y    
end


function crn_dot(vars, subA, subB; max_val=40.0, reltol=1e-8, abstol=1e-8, default=0.0)
    dotss = species(rn_dual_dot)

    # Initial concentration values
    varkeys = [_convert_species2var(sp) for sp in dotss]
    varkeys = [replace(k, "A" => subA) for k in varkeys]
    varkeys = [replace(k, "B" => subB) for k in varkeys]

    uvalues = [get(vars, k, default) for k in varkeys]

    u = Pair.(dotss, uvalues)

    # solve the ODE
    p = []
    sol = simulate_reaction_network(rn_dual_dot, u, p, tspan=(0.0, max_val))

    # Collect the outputs
    yp = sol[end][get_index_of("Yp", dotss)]
    ym = sol[end][get_index_of("Ym", dotss)]
    y = [yp ym]
    return y
end


function crn_dual_node_fwd(rn, vars; tspan=(0.0, 1.0), reltol=1e-4, abstol=1e-6, save_on=false, maxiters=1000)
   
    ss = species(rn)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []
    
    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    # for i in eachindex(u)
    #     println(ss[i], " => ", sol[end][i])
    # end
    
    for i in eachindex(ss)
        if startswith(string(ss[i]), "Z")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "Z", title="CRN | z at t=T |")
end


function crn_main(params, train, val, test; dims=nothing, EPOCHS=10, LR=0.001, 
    tspan=(0.0, 1.0), threshold=0.5, augval=1.0, output_dir="")

    # Initialize a dictionary to track concentrations of all the species
    vars = Dict();

    # Create an output dir and images dir inside it.
    if !isdir("julia/$output_dir")
        mkdir("julia/$output_dir")
        if !isdir("julia/$output_dir/images")
            mkdir("julia/$output_dir/images")
        end
    end

    # Get all the involved CRNs and add their species to the vars
    crns = [rn_dual_node_relu_fwd, rn_dual_node_relu_bwd, rn_param_update, 
            rn_final_layer_update, rn_dissipate_reactions ]
    for crn in crns
        crn_species = species(crn)
        for sp in crn_species
            get!(vars, _convert_species2var(sp), 0.0)
        end
    end

    ## Symbols and what they stand for
    # P: theta
    # B: beta 
    # A: adjoint
    # W: w
    # Z: z
    node_params = copy(params)
    _, theta, beta, w, h, t0, t1 = sequester_params(node_params)

    # It seems like the major axis is off for this. So need to apply transpose. 
    # This is correct. We verified! Trust old rajiv, later rajiv.
    crn_theta = vec(transpose(theta)) 
    # Assign the values of the parameters
    for tindex in eachindex(crn_theta)
        d = _index2Dvar("P", tindex, crn_theta[tindex], dims=dims)
        for (k, v) in d
            vars[k] = v
        end
    end

    # Assign weight parameters
    for windex in eachindex(w)
        d = _index1Dvar("W", windex, w[windex], dims=dims)
        for (k,v) in d
            vars[k] = v
        end
    end

    # Assign h parameter
    hvec = ones(dims)*h
    for hindex in eachindex(hvec)
        d = _index1Dvar("H", hindex, hvec[hindex], dims=dims)
        for (k,v) in d
            vars[k] = v
        end
    end

    # Adding time species, although we don't manipulate them now
    vars["T0"] = t0
    vars["T1"] = t1

    # Assign the beta parameters
    for betaindex in eachindex(beta)
        d = _index1Dvar("B", betaindex, beta[betaindex], dims=dims)
        for (k,v) in d
            vars[k] = v
        end
    end

    ##################### Initialization complete ################
    tr_losses = []
    val_losses = []

    ######### Epoch level Tracking ###########
    crn_tracking = Dictionary()
    for (k, v) in vars
        get!(crn_tracking, k, [])
    end
    get!(crn_tracking, "train_loss", [])
    get!(crn_tracking, "val_loss", [])

    for epoch in 1:EPOCHS
        tr_epoch_loss = 0.0
        for i in eachindex(train)
            println("\n\n========= EPOCH: $epoch | i: $i ===========")
            
            x, y = get_one(train, i)
            x = augment(x, dims-length(x), augval=augval)

            println("-------------------- CRN ---------------------")
            @show x, y
            
            for xindex in eachindex(x)
                d = _index1Dvar("Z", xindex, x[xindex], dims=dims)
                for (k,v) in d
                    vars[k] = v
                end
            end
            
            for xindex in eachindex(x)
                d = _index1Dvar("X", xindex, x[xindex], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

            # Working with classes 0.0 and 1.0
            vars["Yp"] = y
            vars["Ym"] = 0.0
            
            _print_vars(vars, "Z", title="CRN | z at t=0")
            _print_vars(vars, "H", title="CRN | h at t=0")

            # Forward stage
            crn_dual_node_fwd(rn_dual_node_relu_fwd, vars, tspan=tspan)
            
            # Calculate yhat
            yhat = crn_dot(vars, "Z", "W", max_val=40.0)
            @show yhat, yhat[1]-yhat[2]
            
            vars["Op"] = max(0, yhat[1] - yhat[2])
            vars["Om"] = max(0, yhat[2] - yhat[1])
            _print_vars(vars, "O", title="CRN | O at t=T") 
            _print_vars(vars, "Y", title="CRN | Y at t=T")

            # Assigns the vars[Ep] and vars[Em] variables
            crn_create_error_species(vars)
            err = [vars["Ep"] vars["Em"]]
            ###############

            # Epoch loss function
            tr_epoch_loss += 0.5*(err[1]-err[2])^2
            
            # Calculate the output layer gradients
            crn_error_binary_scalar_mult(vars, "Z", "M", max_val=40.0)
            
            # Calculate the adjoint
            crn_error_binary_scalar_mult(vars, "W", "A", max_val=40.0)
            
            #--------------- BACKPROPAGATION BEGIN ----------------#
            
            
            # Backpropagate and calculate parameter gradients 
            crn_dual_backprop(rn_dual_node_relu_bwd, vars, tspan)
            _print_vars(vars, "Z", title="CRN | Z after backprop at t=0 | ")
            _print_vars(vars, "A", title="CRN | A at t=0")
            _print_vars(vars, "G", title="CRN | Gradients at t=0")
            _print_vars(vars, "V", title="CRN | Beta gradients at t=0")
            
            # # Update the final layer weights
            # crn_final_layer_update(vars, LR, (0.0, 40.0))
            _print_vars(vars, "W", title="CRN | Final layer after update |")
            
            # Update the parameters
            crn_param_update(rn_param_update, vars, LR, (0.0, 100.0))
            _print_vars(vars, "P", title="CRN | params after update |")
            _print_vars(vars, "B", title="CRN | beta after update |")
            
            # Tracking parameters
            for (k, v) in vars
                if haskey(crn_tracking, k)
                    push!(crn_tracking[k], v)
                end
            end

            # dissipate_and_annihilate(vars, (0.0, 10.0))
            # _print_vars(vars, "G", title="CRN | Gradients after annihilation")
            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W") || startswith(k, "B") || startswith(k, "H")
                    if endswith(k, "p")
                        m = replace(k, "p"=>"m")
                        tmp = vars[k]-vars[m]
                        vars[k] = max(0, tmp)
                        vars[m] = max(0, -tmp)
                    end
                else
                    vars[k] = 0.0
                end
            end

        end
        tr_epoch_loss /= length(train)
        
        @show epoch, tr_epoch_loss

        # if epoch % 10 != 1
        #     continue
        # end
        ##################################################
        ################# VALIDATION #####################
        val_epoch_loss = 0.0
        val_acc = 0.0

        for i in eachindex(val)
            println("=========VAL EPOCH: $epoch | ITERATION: $i ===========")
            x, y = get_one(val, i)
            x = augment(x, dims - length(x), augval=augval)

            # Working with classes 0.0 and 1.0
            vars["Yp"] = y
            vars["Ym"] = 0.0

            println("===============CRN==========================")
            @show x, y

            for zi in eachindex(x)
                d = _index1Dvar("Z", zi, x[zi], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

            for zi in eachindex(x)
                d = _index1Dvar("X", zi, x[zi], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end
        
            # Forward stage
            crn_dual_node_fwd(rn_dual_node_relu_fwd, vars, tspan=tspan)

            # Calculate yhat            
            yhat = crn_dot(vars, "Z", "W", max_val=40.0)
            @show yhat, yhat[1] - yhat[2]
            val_out = 0.0
            if yhat[1] - yhat[2] >= threshold
                val_out = 1.0
            end
            if val_out == y
                val_acc += 1
            end

            vars["Op"] = max(0, yhat[1] - yhat[2])
            vars["Om"] = max(0, yhat[2] - yhat[1])
            _print_vars(vars, "O", title="CRN | O at t=T")
            _print_vars(vars, "Y", title="CRN | Y at t=T")
            crn_create_error_species(vars)
            err = [vars["Ep"] vars["Em"]]

            # Epoch loss function
            val_epoch_loss += 0.5 * (err[1] - err[2])^2

            # Cancel the dual rail variables to prevent parameters from blowing up
            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W") || startswith(k, "B") || startswith(k, "H")
                    if endswith(k, "p")
                        m = replace(k, "p" => "m")
                        tmp = vars[k] - vars[m]
                        vars[k] = max(0, tmp)
                        vars[m] = max(0, -tmp)
                    end
                else
                    vars[k] = 0.0
                end
            end
        end

        val_epoch_loss /= length(val)
        push!(val_losses, val_epoch_loss)
        push!(tr_losses, tr_epoch_loss)
        val_acc /= length(val)
        @show epoch, val_acc
        # crn_losses_plt = plot([tr_losses, val_losses], label=["train" "val"])
        # png(crn_losses_plt, "julia/$output_dir/images/crn_train_lossplts.png")
        plot()
        myplot([Array(range(1, length(tr_losses))), Array(range(1, length(val_losses)))], [tr_losses, val_losses], ["train_loss", "val_loss"],
            output_dir=output_dir, name="crn_train_lossplts", xlabel="epoch", ylabel="loss")
        plot()
        plot_augmented_state(copy(vars), val, tspan=tspan, dims=dims, threshold=threshold, augval=augval, output_dir=output_dir)
        @show calculate_accuracy(test, copy(vars), tspan=tspan, dims=dims, threshold=threshold, augval=augval, output_dir=output_dir)

        # Plot the tracking parameters.
        plot()
        if !isdir("julia/$output_dir/tracking")
            mkdir("julia/$output_dir/tracking")
        end
        if !isdir("julia/$output_dir/tracking/images")
            mkdir("julia/$output_dir/tracking/images")
        end

        for k in keys(crn_tracking)
            if startswith(k, "P") || startswith(k, "G")
                if endswith(k, "p")
                    pp = k
                    mm = replace(pp, "p"=>"m")
                    values = crn_tracking[pp] .- crn_tracking[mm]
                    kname = replace(k, "p"=>"")
                    plot()
                    myplot([Array(range(1, length(values)))], [values], [kname],
                        output_dir="$output_dir/tracking", name="$kname", xlabel="epoch", ylabel="$kname")
                end
                # savefig("julia/$output_dir/tracking/images/$kname.png")
            end

        end

        
    end
    return vars    
end

function neuralcrn(;DIMS=3)

    open("julia/neuralcrn.log", "w") do fileio  # Write to logs. 
        redirect_stdout(fileio) do 
            ## Linear dataset 
            # train_set = create_linearly_separable_dataset(100, linear, threshold=0.0)
            # val_set = create_linearly_separable_dataset(40, linear, threshold=0.0)
           
            # # Rings 
            # t0 = 0.0
            # t1 = 0.9
            # AUGVAL = 0.2
            # output_dir = "rings_forreproducibility"
            # train = create_annular_rings_dataset(100, lub=0.0, lb=0.45, mb=0.5, ub=1.0)
            # val = create_annular_rings_dataset(200, lub=0.0, lb=0.45, mb=0.5, ub=1.0)
            # test = val

            # Xor dataset
            output_dir  = "xor_forreproducibility"
            t0 = 0.0
            t1 = 0.9
            AUGVAL = 0.2
            train = create_xor_dataset(100)
            val = create_xor_dataset(30)
            test = []
            
            for i in range(0, 100, 40)
                for j in range(0, 100, 40)
                    x1 = i / 100
                    x2 = j / 100
                    x1b = Bool(floor(x1 + 0.5))
                    x2b = Bool(floor(x2 + 0.5))
                    y = Float32(x1b ⊻ x2b)
                    push!(test, [x1 x2 y])
                end
            end
            Random.shuffle!(train)

            # ## AND dataset set t1 = 0.6
            # output_dir = "and_final"
            # t0 = 0.0
            # t1 = 0.9
            # AUGVAL = 0.2
            # train = create_and_dataset(100)
            # val = create_and_dataset(10)    
            # test = []
            # for i in range(0, 100, 40)
            #     for j in range(0, 100, 40)
            #         x1 = i / 100
            #         x2 = j / 100
            #         x1b = Bool(floor(x1 + 0.5))
            #         x2b = Bool(floor(x2 + 0.5))
            #         y = Float32(x1b & x2b)
            #         push!(test, [x1 x2 y])
            #     end
            # end
            # Random.shuffle!(train)

            # # OR dataset
            # output_dir = "or_final"
            # train = create_or_dataset(200)
            # val = create_or_dataset(10)
            # test = []
            # t0 = 0.0
            # t1 = 0.9
            # AUGVAL = 0.2
            # for i in range(0, 100, 40)
            #     for j in range(0, 100, 40)
            #         x1 = i / 100
            #         x2 = j / 100
            #         x1b = Bool(floor(x1 + 0.5))
            #         x2b = Bool(floor(x2 + 0.5))
            #         y = Float32(x1b | x2b)
            #         push!(test, [x1 x2 y])
            #     end
            # end
            # Random.shuffle!(train)

            if !isdir("julia/$output_dir")
                mkdir("julia/$output_dir")
                if !isdir("julia/$output_dir/images")
                    mkdir("julia/$output_dir/images")
                end
            end
            plot()
            myscatter(getindex.(train, 1), getindex.(train, 2), getindex.(train, 3), output_dir=output_dir, name="train",
                xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}")


           
            

            tspan = (t0, t1)
            params_orig = create_node_params(DIMS, t0=t0, t1=t1, h=0.0)
            
            @show params_orig

            println("===============================")
            vars = crn_main(params_orig, train, val, test, EPOCHS=80, dims=DIMS, LR=1.0, tspan=tspan, augval=AUGVAL, output_dir=output_dir)
            @show calculate_accuracy(test, copy(vars), tspan=tspan, dims=DIMS, threshold=0.5, augval=AUGVAL, output_dir=output_dir)
        end
    end
end

neuralcrn(DIMS=3)
#=
Things to do further
1. k_ann = 100.0 in the reactionsReLU for the annihilation reactions. Maybe change it to 10.0

=# 