# This file is used to run CRN operations
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
using IJulia;
using ProgressMeter;
using Distributions;
using Serialization;

include("datasets.jl")
include("utils.jl")
include("relu5D.jl")
include("neuralode.jl")


function _convert_species2var(sp)
    ret = string(sp)
    ret = replace(ret, "(t)" => "")
    return ret
end


function _index2Dvar(sym, index, val; dims1=3, dims2=3)
    first = (index-1) ÷ dims2 
    second = (index-1)% dims2 
    # println(index-1, " ", first, " ", second)
    return Dict( 
        "$(sym)$(first)$(second)p"=> max(0.0, val),
        "$(sym)$(first)$(second)m"=> max(0.0, - val)
    )
end


function _index1Dvar(sym, index, val; dims=3)  
    return Dict(
        "$sym$(index-1)p" => max(0.0, val),
        "$sym$(index-1)m" => max(0.0, -val)
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


function plot_augmented_state(varscopy, dataset; tspan=(0.0, 1.0), dims=3, threshold=0.0, augval=augval)
    aug_x = []
    reg_x = []
    yhats = []
    markers = []

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

        temp = []
        append!(temp, x)
        push!(temp, output)
        push!(yhats, temp)

        
    end
    plt_state1 = scatter3d(getindex.(reg_x, 1), getindex.(reg_x, 2), getindex.(reg_x, 3), group=getindex.(reg_x, 4))
    plt_state2 = scatter3d(getindex.(aug_x, 1), getindex.(aug_x, 2), getindex.(aug_x, 3), group=getindex.(aug_x, 4), markershape=markers) # Color: based on output, shape based on target label. 
    png(plt_state1, "julia/images/crn_before_aug.png")
    png(plt_state2, "julia/images/crn_after_aug.png")
    pltyhats = scatter(getindex.(yhats, 1), getindex.(yhats, 2), group=getindex.(yhats, 4))
    png(pltyhats, "julia/images/crn_yhats.png")
end


function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=3, threshold=0.0, markers=[:circle, :rect], augval=1.0)
    acc = 0
    preds2d = []
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
    end
    # plot()
    # Colors (index = 4) represent the original class the data point belongs to
    # Shapes (index = 3) represent the predicted class of the data point 
    sca = scatter(getindex.(preds2d, 1), getindex.(preds2d, 2), group = getindex.(preds2d, 3)) # output is the label
    png(sca, "julia/images/crn_accuracy_plot.png")
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
    
    for i in eachindex(ss)
        if startswith(string(ss[i]), "Z")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "Z", title="CRN | z at t=T |")
end

function crn_yhat_calculate(rn, vars; tspan=(0.0, 1.0), reltol=1e-4, abstol=1e-6, save_on=false, maxiters=1e4)
    ss = species(rn)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    _print_vars(vars, "W", title="CRN | W at t=T |")
    p = []
    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    for i in eachindex(ss)
        if startswith(string(ss[i]), "O")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "O", title="CRN | Yhat or O at t=T |")
end 

function crn_wgrads_calculate(rn, vars; tspan=(0.0, 1.0), reltol=1e-4, abstol=1e-6, save_on=false, maxiters=1e4)
    ss = species(rn)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []
    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    for i in eachindex(ss)
        if startswith(string(ss[i]), "M")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "M", title="CRN | Wgrads at t=T |")
end

function crn_adjoint_calculate(rn, vars; tspan=(0.0, 1.0), reltol=1e-4, abstol=1e-6, save_on=false, maxiters=1e4)
    ss = species(rn)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []
    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    for i in eachindex(ss)
        if startswith(string(ss[i]), "A")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "A", title="CRN | Adjoint t=T |")
end


function normalize_dataset(dataset; trainmaxs=nothing, trainmins=nothing)
    x, y = dataset[1]
    XSZ = length(x)
    
    maxs = ones(XSZ)*-10000.0
    mins = ones(XSZ)*10000.0
    if trainmaxs === nothing || trainmins === nothing 
        for dindex in eachindex(dataset)
            x, y = dataset[dindex]
            for xi in eachindex(x)
                if x[xi] > maxs[xi]
                    maxs[xi] = x[xi]
                end
                if x[xi] < mins[xi]
                    mins[xi] = x[xi]
                end
            end
        end
    else
        maxs = trainmaxs 
        mins = trainmins
    end
    dataset_norm = []
    for dindex in eachindex(dataset)
        x, y = dataset[dindex]
        x_norm = []
        for xi in eachindex(x)
            push!(x_norm, (x[xi] - mins[xi])/(maxs[xi]-mins[xi] + 1e-6))
        end
        push!(dataset_norm, [x_norm, y])
    end
    return maxs, mins, dataset_norm
end


function crn_main(params, train, val; dims=nothing, EPOCHS=10, LR=0.001, tspan=(0.0, 1.0), 
    threshold=0.5, augval=1.0, num_classes=3, out_dir="julia/images/")
    # Initialize a dictionary to track concentrations of all the species
    vars = Dict();

    # Get all the involved CRNs and add their species to the vars
    crns = [rn_dual_node_relu_fwd, rn_dual_node_relu_bwd, rn_param_update, 
        rn_final_layer_update, rn_yhat_calculate, rn_adjoint_calculate, 
        rn_wgrads_calculate, rn_create_error_species]
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
        d = _index2Dvar("P", tindex, crn_theta[tindex], dims1=dims, dims2=dims)
        for (k, v) in d
            vars[k] = v
        end
    end

    # Assign weight parameters
    for windex in eachindex(w)
        d = _index2Dvar("W", windex, w[windex], dims1=dims, dims2=num_classes)
        print("dd ", d)
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
    val_accs = []

    ######### Epoch level Tracking ###########
    crn_tracking = Dictionary()
    for (k, v) in vars
        get!(crn_tracking, k, [])
    end
    get!(crn_tracking, "train_loss", [])
    get!(crn_tracking, "val_loss", [])
    get!(crn_tracking, "val_acc", [])

    for epoch in 1:EPOCHS
        tr_epoch_loss = 0.0
        for i in eachindex(train)
            println("\n\n========= EPOCH: $epoch | i: $i ===========")
            
            x, y = train[i]
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

            # Assigning y appropriately for multi-class
            for i in range(0, num_classes-1)
                vars["Y$(i)p"] = y[i+1]
                vars["Y$(i)m"] = 0.0
            end
            @show vars
            
            _print_vars(vars, "Z", title="CRN | z at t=0")
            _print_vars(vars, "H", title="CRN | h at t=0")

            # Forward stage
            crn_dual_node_fwd(rn_dual_node_relu_fwd, vars, tspan=tspan)

            # Calculate yhat
            crn_yhat_calculate(rn_yhat_calculate, vars, tspan=(0.0, 40.0))
            # _print_vars(vars, "O", title="CRN | Yhat O at t=T")
            _print_vars(vars, "Y", title="CRN | Y at t=T")

            # Assigns the vars[Ep] and vars[Em] variables
            crn_create_error_species(vars)
            ###############

            # Epoch loss function
            for i in range(0, num_classes-1)
                tr_epoch_loss += 0.5 * (vars["E$(i)p"] - vars["E$(i)m"])^2
            end
            
            # Calculate the output layer gradients
            crn_wgrads_calculate(rn_wgrads_calculate, vars, tspan=(0.0, 100.0))
            # Calculate the adjoint
            crn_adjoint_calculate(rn_adjoint_calculate, vars, tspan=(0.0, 100.0))
            
            #--------------- BACKPROPAGATION BEGIN ----------------#
            
            
            # Backpropagate and calculate parameter gradients 
            crn_dual_backprop(rn_dual_node_relu_bwd, vars, tspan)
            _print_vars(vars, "Z", title="CRN | Z after backprop at t=0 | ")
            _print_vars(vars, "A", title="CRN | A at t=0")
            _print_vars(vars, "G", title="CRN | Gradients at t=0")
            _print_vars(vars, "V", title="CRN | Beta gradients at t=0")
            
            # Update the final layer weights
            crn_final_layer_update(vars, LR, (0.0, 100.0)) # Freezing final layer weights.
            _print_vars(vars, "W", title="CRN | Final layer after update |")
            
            # Update the parameters
            crn_param_update(rn_param_update, vars, LR, (0.0, 100.0))
            _print_vars(vars, "P", title="CRN | params after update |")
            _print_vars(vars, "B", title="CRN | beta after update |")
            
            # Tracking parameters
            for (k, v) in vars
                push!(crn_tracking[k], v)
            end

            
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
        ##################################################
        ################# VALIDATION #####################
    
        val_epoch_loss = 0.0
        val_acc = 0.0

        for i in eachindex(val)
            println("=========VAL EPOCH: $epoch | ITERATION: $i ===========")
            
            x, y = val[i]
            x = augment(x, dims - length(x), augval=augval)
            @show x, y
            
            # Assigning y appropriately for multi-class
            for i in range(0, num_classes - 1)
                vars["Y$(i)p"] = y[i+1]
                vars["Y$(i)m"] = 0.0
            end

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
            crn_yhat_calculate(rn_yhat_calculate, vars, tspan=(0.0, 40.0))

            yhat = []
            for yi in range(0, num_classes-1)
                push!(yhat, vars["O$(yi)p"] - vars["O$(yi)m"])
            end
            if argmax(yhat) == argmax(y)
                val_acc += 1
            end
            @show yhat, y, val_acc, length(val)

            _print_vars(vars, "Y", title="CRN | Y at t=T")

            # Assigns the vars[Ep] and vars[Em] variables
            crn_create_error_species(vars)

            # Epoch loss function
            for i in range(0, num_classes-1)
                val_epoch_loss += 0.5*(vars["E$(i)p"] - vars["E$(i)m"])^2
            end
            


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
        
        push!(crn_tracking["val_loss"], val_epoch_loss)
        push!(crn_tracking["train_loss"], tr_epoch_loss)
        
        val_acc /= length(val)
        @show epoch, val_acc

        push!(crn_tracking["val_acc"], val_acc)
        
        val_accs_plot = plot(crn_tracking["val_acc"], label="val_acc")
        png(val_accs_plot, "$(out_dir)/images/crn_val_accsplt.png")
        
        crn_losses_plt = plot([crn_tracking["train_loss"], crn_tracking["val_loss"]], label=["train" "val"])
        png(crn_losses_plt, "$(out_dir)/images/crn_train_lossplts.png")

        open("$(out_dir)/crn_tracking.pickle", "w") do fileio
            serialize(fileio, crn_tracking)
        end
    end
    return vars    
end

function neuralcrn(; DIMS=5, NUM_CLASSES=3, out_dir="julia/iris")

    open("julia/neuralcrn.log", "w") do fileio  # Write to logs. 
        redirect_stdout(fileio) do 
        
        train, val = create_iris_dataset()
        maxs, mins, train_norm = normalize_dataset(train)
        @show maxs, mins
        _, _, val_norm = normalize_dataset(val, trainmaxs=maxs, trainmins=mins)

        t0 = 0.0
        t1 = 0.6
        AUGVAL = 1.0
        tspan = (t0, t1)
        params_orig = create_node_params(DIMS, t0=t0, t1=t1, h=0.3, num_classes=NUM_CLASSES)
        
        @show params_orig

        println("===============================")
        vars = crn_main(params_orig, train_norm, val_norm, EPOCHS=40, dims=DIMS, LR=1e-2, tspan=tspan, augval=AUGVAL, num_classes=NUM_CLASSES, out_dir=out_dir)
        end
    end
end

neuralcrn(DIMS=5, NUM_CLASSES=3, out_dir="julia/iris")
