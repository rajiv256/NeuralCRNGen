using Pkg; 
# Pkg.add("ReactionNetworkImporters")
# Pkg.add("Dictionaries")
# Pkg.add("LaTeXStrings")
# Pkg.add("Statistics")
# Pkg.add("ColorSchemes")
# Pkg.add("IterTools"); 

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

include("datasets.jl")
include("utils.jl")
include("reactions3D.jl")
include("neuralode.jl")


function _convert_species2var(sp)
    ret = string(sp)
    ret = replace(ret, "(t)" => "")
    return ret
end


# Verified: @show _index2param("P", 3, -3.0)
function _index2Dvar(sym, index, val; dims=3)
    second = (index-1)%dims + 1
    first = (index-1)÷dims + 1
    return Dict(
        "$(sym)$(first)$(second)p"=>max(0, val),
        "$(sym)$(first)$(second)m"=>max(0, -val)
    )
end


function _index1Dvar(sym, index, val; dims=3)
    return Dict(
        "$sym$(index)p"=> max(0.0, val),
        "$sym$(index)m"=> max(0.0, -val)
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
            println("$xvp: $(vars[xvp]) | $xvm: $(vars[xvm]) | $(vars[xvp] - vars[xvm])")
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

############################################################


function plot_augmented_state(varscopy, dataset; tspan=(0.0, 1.0), dims=3)
    aug_x = []
    reg_x = []

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
        

        crn_node_forward(varscopy, (0.0, 1.0), yvec, D=dims)

        yhat = crn_dot(varscopy, "Z", "W", max_val=40.0)
        @show yhat, yhat[1] - yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
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
    png(plt_state1, "before_aug.png")
    png(plt_state2, "after_aug.png")
end


function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=3)
    acc = 0
    epoch_loss = 0.0
    for i in 1:length(dataset)
        x, y = get_one(dataset, i)
        x = augment(x, dims - length(x))

        for zi in eachindex(x)
            d = _index1Dvar("Z", zi, x[zi], dims=dims)
            for (k, v) in d
                varscopy[k] = v
            end
        end

        yvec = [y 1 - y] # threshold could be 0.5 now.
        crn_dual_node_fwd(varscopy, tspan=(0.0, 1.0))
        varscopy["Yp"] = yvec[1]
        varscopy["Ym"] = yvec[2]

        # Calculate yhat            
        yhat = crn_dot(varscopy, "Z", "W", max_val=40.0)
        @show yhat, yhat[1]-yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]
        
        output = 0.0
        if varscopy["Op"] > varscopy["Om"]
            output = 1.0
        end
        if output == y
            acc += 1
        end
    end
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


function crn_param_update(vars, eta, tspan)

    ss = species(rn_param_update)

    u = [vars[_convert_species2var(sp)] for sp in ss]
    
    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1 k2]

    sol = simulate_reaction_network(rn_param_update, u, p, tspan=tspan)
    for i in eachindex(ss)
        if startswith(string(ss[i]), "P")
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


function crn_dual_backprop(vars, tspan; bias=0.01, reltol=1e-8, abstol=1e-8, D=2, default=0.0)


    ss = species(rn_dual_backprop)

    u = [vars[_convert_species2var(sp)] for sp in ss]
    @assert length(u) == length(ss)
    p = []

    sol = simulate_reaction_network(rn_dual_backprop, u, p, tspan=tspan, reltol=reltol, abstol=abstol) # CHECK
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
    # varkeys = [_convert_dotvar(string(sp), subA, subB)  for sp in dotss]
    varkeys = [_convert_species2var(sp) for sp in dotss]
    varkeys = [replace(k, "A" => subA) for k in varkeys]
    varkeys = [replace(k, "B" => subB) for k in varkeys]

    uvalues = [get(vars, k, default) for k in varkeys]

    u = Pair.(dotss, uvalues)

    # solve the ODE
    p = []
    sol = simulate_reaction_network(rn_dual_dot, u, p, tspan=(0.0, max_val), reltol=reltol, abstol=abstol)

    # Collect the outputs
    yp = sol[end][get_index_of("Yp", dotss)]
    ym = sol[end][get_index_of("Ym", dotss)]
    y = [yp ym]
    return y
end


function crn_dual_node_fwd(vars; tspan=(0.0, 1.0))
   
    ss = species(rn_dual_node_fwd)
    u = [vars[_convert_species2var(sp)] for sp in species(rn_dual_node_fwd)]
    p = []
    
    sol = simulate_reaction_network(rn_dual_node_fwd, u, p, tspan=tspan)
    for i in eachindex(u)
        println(ss[i], " => ", sol[end][i])
    end
    
    for i in eachindex(ss)
        if startswith(string(ss[i]), "Z")
            vars[_convert_species2var(ss[i])] = sol[end][i]
        end
    end
    _print_vars(vars, "Z", title="CRN | z at t=T |")
end



function crn_main(params, train, val; dims=nothing, EPOCHS=10, LR=0.001, tspan=(0.0, 1.0))
    # Initialize a dictionary to track concentrations of all the species
    vars = Dict();

    # Get all the involved CRNs and add their species to the vars
    crns = [rn_dual_node_fwd, rn_dual_backprop, rn_param_update, rn_final_layer_update, rn_dissipate_reactions]    
    for crn in crns
        for sp in species(crn)
            get!(vars, _convert_species2var(sp), 0.0)
        end
    end
    
    # Assign the values of the parameters
    for param_index in 1:dims^2
        d = _index2Dvar("P", param_index, params[param_index], dims=dims)
        for (k,v) in d
            vars[k] = v
        end
    end

    # Adding time species, although we don't manipulate them now
    vars["T0"] = 0.0
    vars["T1"] = 1.0

    # Assign the weight parameters
    offset = dims^2 + 2  # 2 for t0 and t1
    for param_index in (offset+1):length(params)
        d = _index1Dvar("W", param_index-offset, params[param_index], dims=dims)
        for (k,v) in d
            vars[k] = v
        end
    end
    
    node_params = copy(params)

    tr_losses = []
    for epoch in 1:EPOCHS
        tr_epoch_loss = 0.0
        for i in eachindex(train)
            println("=================EPOCH: $epoch | ITERATION: $i ==============")
            x, y = get_one(train, i)
            x = augment(x, dims-length(x))
            
            yvec = [y 1-y]

            node_params = one_step_node(x, y, node_params, LR, dims)

            println("===============CRN==========================")
            @show x

            for i in eachindex(x)
                d = _index1Dvar("Z", i, x[i], dims=dims)
                for (k,v) in d
                    vars[k] = v
                end
            end
            vars["Yp"] = yvec[1]
            vars["Ym"] = yvec[2]

            # Forward stage
            crn_dual_node_fwd(vars, tspan=tspan)

            # Calculate yhat            
            yhat = crn_dot(vars, "Z", "W", max_val=40.0)
            @show yhat, yhat[1]-yhat[2]
            vars["Op"] = yhat[1]
            vars["Om"] = yhat[2]

            # Create error species
            err = crn_subtract(yhat, yvec)
            vars["Ep"] = err[1]
            vars["Em"] = err[2]
            _print_vars(vars, "E", title="CRN | Error at t=T")
            # Epoch loss function
            tr_epoch_loss += 0.5*(err[1]-err[2])^2

            
            z = _form_vector(vars, "Z")
            w = _form_vector(vars, "W") 

            # Calculate the gradients of w and the adjoint
            wgrad = vcat([crn_mult(err, z[i, :]) for i in 1:dims])
            wgradsym = _create_symbol_matrix("M", (dims, 1))
            _assign_vars(vars, wgradsym, wgrad)
            _print_vars(vars, "M", title="CRN | Wgrads at t=T")

            # Calculate the adjoint at t=T1
            adj = vcat([crn_mult(err, w[i, :]) for i in 1:dims])
            adjsym = _create_symbol_matrix("A", (dims, 1)) 
            _assign_vars(vars, adjsym, adj)
            _print_vars(vars, "A", title="CRN | Adjoint at t=T")
            
            # Backpropagate and calculate parameter gradients 
            crn_dual_backprop(vars, tspan)
            _print_vars(vars, "G", title="CRN | Gradients at t=0")
            

            # Update the final layer Weights
            crn_final_layer_update(vars, LR, (0.0, 100.0))
            _print_vars(vars, "W", title="CRN | Final layer at t=0|")
            # Update the parameters
            crn_param_update(vars, LR, (0.0, 100.0))
            _print_vars(vars, "P", title="CRN | Parameters at t=0|")
            

            # dissipate_and_annihilate(vars, (0.0, 10.0))
            # _print_vars(vars, "G", title="CRN | Gradients after annihilation")
            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W")
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
        push!(tr_losses, tr_epoch_loss)
        trplt = plot(tr_losses)
        png(trplt, "training_losses.png")
    end
    return vars    
end

function neuralcrn(;DIMS=3)
    # train = create_linearly_separable_dataset(100, linear, threshold=0.0)
    # val = create_linearly_separable_dataset(40, linear, threshold=0.0)
    train = create_annular_rings_dataset(100, 1.0)
    val = create_annular_rings_dataset(40, 1.0)

    params_orig = create_node_params(DIMS, t0=0.0, t1=1.0)
    
    println("===============================")
    vars = crn_main(params_orig, train, val, EPOCHS=20, dims=DIMS, tspan=(0.0, 1.0))
    
    @show calculate_accuracy(val, copy(vars), dims=3)
    plot_augmented_state(copy(vars), val)
end

neuralcrn(DIMS=3)

# _filter_rn_species(rn_dual_node_fwd, prefix="Z")
