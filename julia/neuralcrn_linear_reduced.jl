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
# include("reactions2D.jl")
include("linear_reduced2D.jl")
# include("neuralode.jl")
include("myplots.jl")


function _convert_species2var(sp)
    ret = string(sp)
    ret = replace(ret, "(t)" => "")
    return ret
end


# # Verified: @show _index2param("P", 3, -3.0)
# function _index2Dvar(sym, index, val; dims=2)
#     second = (index - 1) % dims + 1
#     first = (index - 1) ÷ dims + 1
#     return Dict(
#         "$(sym)$(first)$(second)p" => max(0, val),
#         "$(sym)$(first)$(second)m" => max(0, -val)
#     )
# end


function _index1Dvar(sym, index, val; dims=2)
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
            xvm = replace(xvp, "p" => "m")
            println("$xvp: $(vars[xvp]) | $xvm: $(vars[xvm]) | $(vars[xvp] - vars[xvm])")
        end
    end
    println("---------------------")
end


function _filter_rn_species(ss; prefix="Z")
    xs = filter(x -> startswith(string(x), prefix), ss)
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
function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=2, output_dir="linear_reduced", threshold=2.0, pos=4.0, neg=0.0, tag="")
    acc = 0
    targets = []
    outputs = []
    xs = []
    ys = []

    wrongs = [] # (list of [x1 x2])

    
    for i in 1:length(dataset)
        x, y = get_one(dataset, i)
        x = augment(x, dims - length(x))

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

        yvec = [y 0]
        crn_dual_node_fwd(varscopy, tspan=tspan)
        varscopy["Yp"] = yvec[1]
        varscopy["Ym"] = yvec[2]

        # Calculate yhat            
        yhat = crn_dot(varscopy, "Z", "W", max_val=40.0)
        @show yhat, yhat[1] - yhat[2]
        varscopy["Op"] = yhat[1]
        varscopy["Om"] = yhat[2]

        push!(targets, y)
        push!(xs, x[1])
        push!(ys, x[2])

        if yhat[1] - yhat[2] > threshold
            push!(outputs, pos)
            @show yhat[1], yhat[2], y, pos, neg
            if y != neg
                acc += 1
            else
                push!(wrongs, x)
            end
        end
        if yhat[1] - yhat[2] <= threshold
            push!(outputs, neg)
            if y == neg
                acc += 1
            else
                push!(wrongs, x)
            end
        end

    end
    plot()
    myscatter(xs, ys, outputs, output_dir=output_dir, name="$(tag)_outputs", xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}", markershape=:square)
    plot()
    gg = myscatter(xs, ys, outputs, output_dir=output_dir, name="$(tag)_outputs", markersize=4, markershape=:square)
    gg = myscatternogroup(getindex.(wrongs, 1), getindex.(wrongs, 2), markershape=:xcross, markercolor="black", markersize=4, label="errors", output_dir=output_dir, name="$(tag)_outputs_with_wrongs", xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}")
    # gg = scatter!(getindex.(wrongs, 1), getindex.(wrongs, 2), markershape=:xcross, markercolor="black", markersize=4, label="errors",
    #     xtickfontsize=12, ytickfontsize=12,
    #     legendfontsize=12, fontfamily="Arial", grid=false,
    #     framestyle=:semi, widen=false)
    # savefig(gg, "julia/$output_dir/images/$(tag)_outputs_with_wrongs.svg")
    # savefig(gg, "julia/$output_dir/images/$(tag)_outputs_with_wrongs.png")
    return acc / length(dataset)
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


    ss = species(rn_dual_node_bwd)

    u = [vars[_convert_species2var(sp)] for sp in ss]
    @assert length(u) == length(ss)
    p = []

    sol = simulate_reaction_network(rn_dual_node_bwd, u, p, tspan=tspan, reltol=reltol, abstol=abstol) # CHECK
    for i in eachindex(ss)
        vars[_convert_species2var(ss[i])] = sol[end][i]
    end
end


function crn_subtract(a, b; max_val=40.0, default=0.0)
    u = [
        :Ap => a[1], :Am => a[2], :Bp => b[1], :Bm => b[2],
        :Yp => 0.0, :Ym => 0.0
    ]
    p = []
    sol = simulate_reaction_network(rn_dual_subtract, u, p, tspan=(0.0, max_val))

    ss = species(rn_dual_subtract)
    yp = sol[end][get_index_of("Yp", ss)]
    ym = sol[end][get_index_of("Ym", ss)]
    y = [yp ym]
    println(a, b, y, "=======")
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


function crn_main(params, train, val; dims=2, EPOCHS=10, LR=0.01, tspan=(0.0, 1.0), output_dir="linear_reduced", threshold=2.0, pos=0.0, neg=4.0)
    # Initialize a dictionary to track concentrations of all the species
    vars = Dict()

    # Get all the involved CRNs and add their species to the vars
    crns = [rn_dual_node_fwd, rn_dual_node_bwd, rn_param_update, rn_final_layer_update, rn_dissipate_reactions]
    for crn in crns
        for sp in species(crn)
            get!(vars, _convert_species2var(sp), 0.0)
        end
    end

    # Assign the values of the parameters
    for param_index in 1:dims
        d = _index1Dvar("P", param_index, params[param_index], dims=dims)
        for (k, v) in d
            vars[k] = v
        end
    end

    # Adding time species, although we don't manipulate them now
    vars["T0"] = getindex(tspan, 1)
    vars["T1"] = getindex(tspan, 2)

    # Assign the weight parameters
    offset = dims + 2  # 2 for t0 and t1
    for param_index in (offset+1):(offset + dims)
        d = _index1Dvar("W", param_index - offset, params[param_index], dims=dims)
        for (k, v) in d
            vars[k] = v
        end
    end
    
    offset = dims + 2 + dims
    d = _index1Dvar("H", 1, params[offset + 1], dims=dims)
    for (k, v) in d
        vars[k] = v
    end
    d = _index1Dvar("H", 2, params[offset + 1], dims=dims)
    for (k, v) in d
        vars[k] = v
    end


    tr_losses = []
    val_losses = []
    val_accs = []
    for epoch in 1:EPOCHS
        tr_epoch_loss = 0.0
        

        ##### VALIDATION ###################################
        ####################################################

        val_epoch_loss = 0.0    
        val_acc = 0.0 
        val_outs = []
        for i in eachindex(val)
            x, y = get_one(val, i)
            x = augment(x, dims - length(x))

            yvec = [y 0]
            
            println("===============CRN==========================")
            @show x

            for i in eachindex(x)
                d = _index1Dvar("Z", i, x[i], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end
            for i in eachindex(x)
                d = _index1Dvar("X", i, x[i], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

            vars["Yp"] = yvec[1]
            vars["Ym"] = yvec[2]

            # Forward stage
            crn_dual_node_fwd(vars, tspan=tspan)

            # Calculate yhat            
            yhat = crn_dot(vars, "Z", "W", max_val=40.0)
            @show yhat, yhat[1] - yhat[2]
            
            push!(val_outs, [yhat[1]-yhat[2], y])
            vars["Op"] = yhat[1]
            vars["Om"] = yhat[2]


            # Create error species
            err = crn_subtract(yhat, yvec)
            vars["Ep"] = err[1]
            vars["Em"] = err[2]
            _print_vars(vars, "E", title="CRN | Error at t=T")

            # Epoch loss function
            val_epoch_loss += 0.5 * (err[1] - err[2])^2
            # val_epoch_loss += abs(err[1] - err[2])

            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W") || startswith(k, "H")
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


        for i in eachindex(train)
            println("=================EPOCH: $epoch | ITERATION: $i ==============")
            x, y = get_one(train, i)
            x = augment(x, dims - length(x))

            yvec = [y 0] # 14 May 2024 change.

            println("===============CRN==========================")
            @show x

            for i in eachindex(x)
                d = _index1Dvar("Z", i, x[i], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

            for i in eachindex(x)
                d = _index1Dvar("X", i, x[i], dims=dims)
                for (k, v) in d
                    vars[k] = v
                end
            end

            vars["Yp"] = yvec[1]
            vars["Ym"] = yvec[2]
            _print_vars(vars, "Z", title="CRN | Z at t = 0")
            # Forward stage
            crn_dual_node_fwd(vars, tspan=tspan)
            _print_vars(vars, "Z", title="CRN | Z at t = T")
            # Calculate yhat            
            yhat = crn_dot(vars, "Z", "W", max_val=40.0)
            @show yhat, yhat[1] - yhat[2]
            vars["Op"] = yhat[1]
            vars["Om"] = yhat[2]

            # Create error species
            err = crn_subtract(yhat, yvec)
            vars["Ep"] = err[1]
            vars["Em"] = err[2]
            _print_vars(vars, "E", title="CRN | Error at t=T")

            # Epoch loss function
            tr_epoch_loss += 0.5 * (err[1] - err[2])^2
            # tr_epoch_loss += abs(err[1] - err[2])


            z = _form_vector(vars, "Z")
            w = _form_vector(vars, "W")

            
            # Calculate the adjoint at t=T1
            adj = vcat([crn_mult(err, w[i, :]) for i in 1:dims])
            adjsym = _create_symbol_matrix("A", (dims, 1))
            _assign_vars(vars, adjsym, adj)

            # Backpropagate and calculate parameter gradients 
            _print_vars(vars, "G", title="CRN | Gradients at t=T")
            crn_dual_backprop(vars, tspan)
            _print_vars(vars, "G", title="CRN | Gradients at t=0")
            _print_vars(vars, "A", title="CRN | Adjoint at t=0")

            # Update the final layer Weights
            # crn_final_layer_update(vars, LR, (0.0, 100.0))
            _print_vars(vars, "W", title="CRN | Final layer at t=0|")
            # Update the parameters
            _print_vars(vars,  "P", title="CRN | Parameters before update")
            # crn_param_update(vars, LR, (0.0, 30.0))
            vars["P1p"] = vars["P1p"] + LR * vars["G1m"]
            vars["P1m"] = vars["P1m"] + LR * vars["G1p"]
            vars["P2p"] = vars["P2p"] + LR * vars["G2m"]
            vars["P2m"] = vars["P2m"] + LR * vars["G2p"]


            _print_vars(vars, "P", title="CRN | Parameters after update")


            for k in keys(vars)
                if startswith(k, "P") || startswith(k, "W") || startswith(k, "H")
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


        
        tr_epoch_loss /= length(train)
        
        val_epoch_loss /= length(val)
        val_acc /= length(val)
        @show epoch, val_acc
        push!(tr_losses, tr_epoch_loss)
        push!(val_losses, val_epoch_loss)
        push!(val_accs, val_acc)
        
        # trplt = plot(tr_losses)
        # # png(trplt, "training_losses.png")
        plot()
        myplot([Array(range(1, epoch)), Array(range(1, epoch))], [tr_losses, val_losses], ["train_loss", "val_loss"],
            output_dir=output_dir, name="training_losses", xlabel="epoch", ylabel="loss")
        plot()
        # myscatternogroup(val_yhats, val_ys, output_dir=output_dir, name="val_comparision", label="comparision", xlabel="predicted", ylabel="target")
        gg = scatter!(getindex.(val_outs, 1), getindex.(val_outs, 2), xlabel="Predicted", ylabel="Target")
        savefig(gg, "julia/$(output_dir)/images/val_comparision.png")
        # myplot([Array(range(1, epoch))], [val_accs], ["val_acc"],
        #     output_dir=output_dir, name="val_accuracies", xlabel="epoch", ylabel="accuracy")
        # plot()
        # @show calculate_accuracy(val, copy(vars), dims=2, output_dir=output_dir, threshold=threshold, pos=neg, neg=neg, tag="val")
    end
    return vars
end


function neuralcrn(; DIMS=2, output_dir="linear_reduced")
    THRESHOLD = 2.0
    POS = 4.0
    NEG = 0.0
    MINI = 0.0
    MAXI = THRESHOLD
    LR = 0.01
    EPOCHS = 10
    TSPAN = (0.0, 0.6)
    T0 = 0.0
    T1 = 1.0
    BIAS = 0.0

    train = create_linear_regression_dataset(100, linear_reduced, mini=MINI, maxi=MAXI)
    val = create_linear_regression_dataset(100, linear_reduced, mini=MINI, maxi=MAXI)
    test = []
    
    # for i in range(0, Int32(THRESHOLD*100), 60)
    #     for j in range(0, Int32(THRESHOLD*100), 60)

    #         x1 = i / 100
    #         x2 = j / 100
    #         y = NEG
    #         yval = linear_reduced(x1, x2)
    #         if yval > THRESHOLD
    #             y = POS
    #         end
    #         push!(test, [x1; x2; y])
    #     end
    # end
    if !isdir("julia/$output_dir")
        mkdir("julia/$output_dir")
        if !isdir("julia/$output_dir/images")
            mkdir("julia/$output_dir/images")
        end
    end
    # print(getindex.(train, 3))
    # myscatter(getindex.(train, 1), getindex.(train, 2), getindex.(train, 3), output_dir=output_dir, name="train",
    #     xlabel=L"\mathbf{\mathrm{x_1}}", ylabel=L"\mathbf{\mathrm{x_2}}")

    params_orig = create_node_params_reduced(DIMS, t0=T0, t1=T1, h=BIAS)
    open("julia/neuralcrn.log", "w") do fileio  # Write to logs. 
        redirect_stdout(fileio) do
            println("===============================")
            vars = crn_main(params_orig, train, val, EPOCHS=EPOCHS, tspan=TSPAN, output_dir=output_dir, LR=LR, threshold=THRESHOLD, pos=POS, neg=NEG)
            plot()
            
        end
    end
end

neuralcrn(output_dir="linear_regression")

# _filter_rn_species(rn_dual_node_fwd, prefix="Z")
