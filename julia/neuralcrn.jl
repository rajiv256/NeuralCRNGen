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

include("datasets.jl")
include("utils.jl")
include("linear_reduced3D.jl")


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
    # # @show varkeys
    u = [get(vars, k, 0.0) for k in varkeys]
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
    # _print_vars(vars, "O", title="CRN | Output after annihilation at t=T")
end


function crn_create_error_species(vars)
    # CHECK: Assuming that Y and O species are up to date
    ss = species(rn_create_error_species)
    # # @show ss 
    # # @show vars
    u = [vars[_convert_species2var(sp)] for sp in ss]
    # # @show u
    p = []
    # Due to the way in which `rn_create_error_species` is setup, the tspan has to be (0.0, 1.0)
    sol = simulate_reaction_network(rn_create_error_species, u, p, tspan=(0.0,1.0))
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
    _print_vars(vars, "E", title="CRN | Error species at t=T")
end


function plot_augmented_state(varscopy, dataset; tspan=(0.0, 1.0), dims=3, threshold=0.0)
    aug_x = []
    reg_x = []
    yhats = []

    for i in eachindex(dataset)
        x, y = get_one(dataset, i)

        x = augment(x, dims - length(x))
        yvec = [y 1 - y]
        for zi in eachindex(x)
            d = _index1Dvar("Z", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end
        varscopy["Yp"] = yvec[1]
        varscopy["Ym"] = yvec[2]
        elem = []
        append!(elem, x)
        push!(elem, y)
        push!(reg_x, elem)
        
        crn_dual_node_fwd(rn_dual_node_relu_fwd, varscopy, tspan=(0.0, 1.0))

        yhat = crn_dot(rn_dual_dot, varscopy, "Z", "W", max_val=40.0)
        # # @show yhat, yhat[1] - yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
        if yhat[1]-yhat[2] >= threshold
            exp = 1.0
        else
            exp = 0.0
        end
        temp = []
        append!(temp, x)
        push!(temp, exp)
        push!(yhats, temp)

        push!(
            aug_x, [
                varscopy["Z1p"] - varscopy["Z1m"],
                varscopy["Z2p"] - varscopy["Z2m"],
                varscopy["Z3p"] - varscopy["Z3m"],
                yvec[1] - yvec[2]
            ]
        )
    end
    plt_state1 = scatter3d(getindex.(reg_x, 1), getindex.(reg_x, 2), getindex.(reg_x, 3), group=getindex.(reg_x, 4))
    plt_state2 = scatter3d(getindex.(aug_x, 1), getindex.(aug_x, 2), getindex.(aug_x, 3), group=getindex.(aug_x, 4))
    png(plt_state1, "julia/images/crn_before_aug.png")
    png(plt_state2, "julia/images/crn_after_aug.png")
    pltyhats = scatter3d(getindex.(yhats, 1), getindex.(yhats, 2), getindex.(yhats, 3), group=getindex.(yhats, 4))
    png(pltyhats, "julia/images/crn_yhats.png")
end


function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=3, threshold=0.5, markers=[:circle, :rect], neg=0.0, pos=1.0)
    acc = 0
    preds2d = []
    for i in 1:length(dataset)
        x, y = get_one(dataset, i)
        
        temp = []
        append!(temp, x)

        x = augment(x, dims - length(x))

        for zi in eachindex(x)
            d = _index1Dvar("Z", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end

        
        crn_dual_node_fwd(rn_dual_node_relu_fwd, varscopy, tspan=tspan)
        varscopy["Yp"] = max(0.0, y)
        varscopy["Ym"] = max(0.0, -y)

        # Calculate yhat            
        yhat = crn_dot(rn_dual_dot, varscopy, "Z", "W", max_val=40.0)
        # # @show yhat, yhat[1]-yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
        output = neg
        if varscopy["Op"] - varscopy["Om"] >= threshold # TODO: CHECK BEFORE
            output = pos
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
    sca = scatter(getindex.(preds2d, 1), getindex.(preds2d, 2), group=getindex.(preds2d, 3), markershape=:circ, markersize=4)
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
end


function crn_final_layer_update(vars, eta, tspan)
    ss = species(rn_final_layer_update)
    # # @show ss
    u = [vars[_convert_species2var(sp)] for sp in ss]
    # # @show u
    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1 k2]
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
    sol = simulate_reaction_network(rn, u, p, tspan=tspan)
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
end


function crn_subtract(a, b; max_val=40.0, default=0.0)
    u = [
        :Jp => a[1], :Jm => a[2], :Kp => b[1], :Km => b[2],
        :Lp => default, :Lm => default
    ]
    p = []
    sol = simulate_reaction_network(rn_dual_subtract, u, p, tspan=(0.0, max_val))

    ss = species(rn_dual_subtract)
    yp = sol[end][get_index_of("Lp", ss)]
    ym = sol[end][get_index_of("Lm", ss)]
    y = [yp ym]
    return y
end


function crn_mult(a, b; max_val=100.0)
    # a: [ap, am]   b: [bp, bm]

    u = [:Jp => a[1], :Jm => a[2], :Kp => b[1], 
        :Km => b[2], :Lp => 0, :Lm => 0]
    p = []
    sol = simulate_reaction_network(rn_dual_mult, u, p, tspan=(0.0, max_val))

    # Calculate value
    ss = species(rn_dual_mult)

    yp = sol[end][get_index_of("Lp", ss)]
    ym = sol[end][get_index_of("Lm", ss)]
    y = [yp ym]
    return y    
end


function crn_dot(rn, vars, subJ, subK; max_val=40.0, reltol=1e-8, abstol=1e-8, default=0.0)
    # # @show vars
    dotss = species(rn)
    # Initial concentration values
    varkeys = [_convert_species2var(sp) for sp in dotss]
    varkeys = [replace(k, "A" => subJ) for k in varkeys]
    varkeys = [replace(k, "B" => subK) for k in varkeys]
    uvalues = [get(vars, k, default) for k in varkeys]
    # # @show varkeys
    # # @show uvalues
    u = uvalues
    p = []
    sol = simulate_reaction_network(rn, u, p, tspan=(0, max_val))

    # Collect the outputs
    yhatp = sol[end][get_index_of("Yp", dotss)]
    yhatm = sol[end][get_index_of("Ym", dotss)]
    yhat = [yhatp yhatm]
    return yhat
end


function crn_dual_node_fwd(rn, vars; tspan=(0.0, 1.0), reltol=1e-4, abstol=1e-6, save_on=false)
   
    ss = species(rn)
    u = [vars[_convert_species2var(sp)] for sp in ss]
    p = []
    
    sol = simulate_reaction_network(rn, u, p, tspan=tspan, save_on=save_on)
    
    for i in eachindex(ss)
        if startswith(string(ss[i]), "Z")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
end


function crn_main(params, train, val; dims=nothing, EPOCHS=10, LR=1.0, tspan=(0.0, 1.0), pos=1.0, neg=0.0, threshold=0.5, CLIPGRAD=1000.0, augval=0.8)

    # Initialize a dictionary to track concentrations of all the species
    vars = Dict();

    # Get all the involved CRNs and add their species to the vars
    crns = [rn_dual_node_relu_fwd, rn_dual_node_relu_bwd, rn_param_update, rn_create_error_species]
    for crn in crns
        crn_species = species(crn)
        for sp in crn_species
            get!(vars, _convert_species2var(sp), 0.0)
        end
    end

    node_params = copy(params)
    _, h, theta, w, t0, t1 = sequester_params(node_params)

    # It seems like the major axis is off for this. So need to apply transpose. 
    # This is correct. We verified! Trust old rajiv, later rajiv.
    crn_theta = vec(transpose(theta)) 
    # Assign the values of the parameters
    for tindex in eachindex(crn_theta)
        d = _index1Dvar("P", tindex, crn_theta[tindex], dims=dims)
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

    # Adding time species, although we don't manipulate them now
    vars["T0"] = t0
    vars["T1"] = t1

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
            

            if y <= 0.0
                vars["Yp"] = 0.0
                vars["Ym"] = -y
            else
                vars["Yp"] = y
                vars["Ym"] = 0.0
            end

            println("---- y : ", y)
            _print_vars(vars, "Y", title="CRN | Y at t=T")
            _print_vars(vars, "Z", title="CRN | z at t=0")
            _print_vars(vars, "P", title="CRN | p at t=0")

            # Forward stage
            crn_dual_node_fwd(rn_dual_node_relu_fwd, vars, tspan=tspan)
            _print_vars(vars, "Z", title="CRN | Z at t=T")
            _print_vars(vars, "W", title="CRN | W at t=T")

            # Calculate yhat
            yhat = crn_dot(rn_dual_dot, vars, "Z", "W", max_val=40.0)
            yhatval = yhat[1] - yhat[2]  
            @show yhatval

            if yhatval <= 0.0
                vars["Op"] = 0.0
                vars["Om"] = -yhatval 
            else 
                vars["Op"] = yhatval
                vars["Om"] = 0.0
            end

            _print_vars(vars, "O", title="CRN | O at t=T") 
            _print_vars(vars, "Y", title="CRN | Y at t=T")

            # Assigns the vars[Ep] and vars[Em] variables
            crn_create_error_species(vars)
            err = [vars["Ep"] vars["Em"]]
            ###############

            # Epoch loss function
            tr_epoch_loss += 0.5*(err[1]-err[2])^2
            
            # Calculate the output layer gradients
            # crn_error_binary_scalar_mult(vars, "Z", "M", max_val=40.0)
             
            # Calculate the adjoint
            crn_error_binary_scalar_mult(vars, "W", "A", max_val=40.0)
            
            println("-------BACKPROP-------")
            
            # Backpropagate and calculate parameter gradients 
            crn_dual_backprop(rn_dual_node_relu_bwd, vars, tspan)

            # Clip the gradient values
            for k in keys(vars)
                if startswith(k, "G") || startswith(k, "M")
                    vars[k] = min(vars[k], CLIPGRAD)
                end
            end
            _print_vars(vars, "Z", title="CRN | Z after backprop at t=0 | ")
            _print_vars(vars, "A", title="CRN | A at t=0")
            _print_vars(vars, "G", title="CRN | Gradients at t=0")

            # Tracking parameters
            for (k, v) in vars
                push!(crn_tracking[k], v)
            end

            for k in keys(vars)
                if startswith(k, "G")
                    if vars[k] > CLIPGRAD
                        for k in keys(vars)
                            if startswith(k, "P") || startswith(k, "W")
                                if endswith(k, "p")
                                    m = replace(k, "p" => "m")
                                    tmp = vars[k] - vars[m]
                                    vars[k] = max(0, tmp)
                                    vars[m] = max(0, -tmp)
                                end
                            else
                                # Other than the above mentioned parameters 
                                vars[k] = 0.0
                            end
                        end
                        continue
                    end
                end
            end


            # Update the final layer weights
            # crn_final_layer_update(vars, LR, (0.0, 40.0))

            # Update the parameters
            crn_param_update(rn_param_update, vars, LR, (0.0, 40.0))
            # if i == 2
            #     return 
            # end

            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W")
                    if endswith(k, "p")
                        m = replace(k, "p"=>"m")
                        tmp = vars[k]-vars[m]
                        vars[k] = max(0, tmp)
                        vars[m] = max(0, -tmp)
                    end
                else
                    # Other than the above mentioned parameters 
                    vars[k] = 0.0
                end
            end
        end
        
        tr_epoch_loss /= length(train)
        # @show tr_epoch_loss

        if epoch % 2 != 0
            continue
        end

        ###################################################
        ################## VALIDATION #####################
        
        val_epoch_loss = 0.0
        val_acc = 0.0
        for i in eachindex(val)
            println("=========VAL EPOCH: $epoch | ITERATION: $i ===========")
            x, y = get_one(val, i)
            x = augment(x, dims - length(x), augval=augval)

            println("===============CRN==========================")
            # # @show x, y

            for zi in eachindex(x)
                d = _index1Dvar("Z", zi, x[zi], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end
            for xindex in eachindex(x)
                d = _index1Dvar("X", xindex, x[xindex], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

        
            # Forward stage
            crn_dual_node_fwd(rn_dual_node_relu_fwd, vars, tspan=tspan)

            # Calculate yhat            
            yhat = crn_dot(rn_dual_dot, vars, "Z", "W", max_val=40.0)
            # # @show yhat, yhat[1] - yhat[2]
            val_out = neg
            if yhat[1] - yhat[2] >= threshold
                val_out = pos
            end
            if val_out == y
                val_acc += 1
            end

            yhatval = yhat[1] - yhat[2]
            if yhatval <= 0.0
                vars["Op"] = 0.0
                vars["Om"] = -yhatval
            else
                vars["Op"] = yhatval
                vars["Om"] = 0.0
            end
            
            # _print_vars(vars, "O", title="CRN | O at t=T")
            

            if y <= 0.0
                vars["Yp"] = 0.0
                vars["Ym"] = -y 
            else 
                vars["Yp"] = y
                vars["Ym"] = 0.0
            end

            println("---- y : ", y)
            # _print_vars(vars, "O", title="CRN | O at t=T")
            crn_create_error_species(vars)
            err = [vars["Ep"] vars["Em"]]
                

            # Epoch loss function
            val_epoch_loss += 0.5 * (err[1] - err[2])^2

            # Cancel the dual rail variables to prevent parameters from blowing up
            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W")
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
        # @show epoch, val_acc
        crn_losses_plt = plot([tr_losses, val_losses], label=["train" "val"])
        png(crn_losses_plt, "julia/images/crn_train_lossplts.png")
        
        # Plot tracking information
        for k in keys(vars)
            if startswith(k, "P") || startswith(k, "W") || startswith(k, "G")
                if endswith(k, "p")
                    m = replace(k, "p" => "m")
                    diffarr = crn_tracking[k] - crn_tracking[m]
                    name = replace(k, "p" => "")
                    gg = plot(diffarr, label=name)
                    png(gg, "julia/images/tracking/$name.png")
                end
            end
        end
            
        calculate_accuracy(val, copy(vars), tspan=tspan, dims=dims, pos=pos, neg=neg, threshold=threshold)       
         
    end
    return vars    
end

function neuralcrn(;DIMS=3)

    open("julia/neuralcrn.log", "w") do fileio  # Write to logs. 
        redirect_stdout(fileio) do    
        POS = 1.0
        NEG = 0.0
        THRESHOLD = 0.5
        # train = create_linearly_separable_dataset(100, linear, threshold=1.0)
        # val = create_linearly_separable_dataset(100, linear, threshold=1.0)
        # Rings 
        # train = create_annular_rings_dataset(100, lub=0.0, lb=0.5, mb=0.55, ub=1.0)
        # val = create_annular_rings_dataset(80, lub=0.0, lb=0.5, mb=0.55, ub=1.0)

        train = create_xor_dataset(100)
        val = create_xor_dataset(300)
        t0 = 0.0
        t1 = 1.0
        tspan = (t0, t1)
        params_orig = create_node_params(DIMS, t0=t0, t1=t1)
        params_orig_copy = copy(params_orig)
        # @show params_orig_copy
        println("===============================", params_orig)
        vars = crn_main(params_orig, train, val, EPOCHS=100, dims=DIMS, LR=1, tspan=tspan, pos=POS, neg=NEG, threshold=THRESHOLD)
        end
    end

end

neuralcrn(DIMS=3)
#=
Things to do further
1. k_ann = 100.0 in the reactionsReLU for the annihilation reactions. Maybe change it to 10.0

=# 