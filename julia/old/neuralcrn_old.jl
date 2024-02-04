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
include("reactions2D.jl")



# simulates multiplication of two scalars
function mult(a, b; max_val=40.0)
    # a: [ap, am]   b: [bp, bm]

    u = [:Ap => a[1], :Am => a[2], :Bp => b[1], :Bm => b[2], :Yp => 0, :Ym => 0]
    p = []
    sol = simulate_reaction_network(rn_dual_mult, u, p, tspan=(0.0, max_val))

    # Calculate value
    ss = species(rn_dual_mult)

    yp = sol[end][get_index_of("Yp", ss)]
    ym = sol[end][get_index_of("Ym", ss)]
    y = [yp ym]
    return y    
end


function _generate_u(vars, rn; default=0.0)
    ss = species(rn)
    uvalues = [get!(vars, _convert_species2var(sp), default) for sp in ss]
    u = Pair.(ss, uvalues)
    return u 
end


function _convert_dotvar(s, subA, subB)
     ret = s 
     ret = replace(ret, "(t)" => "")
     ret = replace(ret, "A" => subA)
     ret = replace(ret, "B" => subB)
     return ret 
end


function subtract(ap, am, bp, bm; max_val=40.0, default=0.0)
    u = [
        :Ap => ap, :Am => am, :Bp => bp, :Bm => bm, 
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


function dot(vars, subA, subB; max_val=40.0, reltol=1e-8, abstol=1e-8, default=0.0)
    dotss = species(rn_dual_dot)

    # Initial concentration values
    # varkeys = [_convert_dotvar(string(sp), subA, subB)  for sp in dotss]
    varkeys = [_convert_species2var(sp) for sp in dotss]
    varkeys = [replace(k, "A"=>subA) for k in varkeys]
    varkeys = [replace(k, "B"=>subB) for k in varkeys]

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


function convert_to_dual_rail(x)
    return (max(0, x), abs(min(0, x)))
end


# Converts the outputs from scalars to binary species.
function crn_output_annihilation(vars, tspan)
    # Note: No need of variable transformation here
    ss = species(rn_output_annihilation)
    uvalues = [get!(vars, _convert_species2var(sp), 0.0) for sp in ss]
    u = Pair.(ss, uvalues)
    p = []
    sol = simulate_reaction_network(rn_output_annihilation, u, p, tspan=tspan)
    return sol
    # Update the output values

end


function crn_create_error_species(vars, tspan)
    u = _generate_u(vars, rn_create_error_species)
    p = []
    sol = simulate_reaction_network(rn_create_error_species, u, p, tspan=tspan)
    return sol
    
end


function crn_node_forward(vars, tspan, y; precision=4, D=2)
    # println("============= CRN FORWARD STEP =====================")
    # Get the species in the order in which the function `species` retrieves them
    sps = get_species_array(rn_dual_node_fwd)
    
    uvalues = [vars[sp] for sp in sps]
    # Assigning initial values
    u = Pair.(species(rn_dual_node_fwd), uvalues)
    p = []  # All the reactions have unit rate constant, stays empty
    
    sol = simulate_reaction_network(rn_dual_node_fwd, u, p, tspan=tspan)
    return sol 
    # println("============= END CRN FORWARD STEP ===============")
end


function _print_vars(vars, prefix; title="")
    println("Title: ", title, " | Prefix: ", prefix)
    Pspecies = [k for k in keys(vars) if startswith(k, prefix)]
    sort!(Pspecies)
    for psp in Pspecies
        print(psp, ": ", vars[psp], " | ")
    end
    println("-------------")
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
            continue; 
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
    

function _convert_species2var(sp)
    ret = string(sp)
    ret = replace(ret, "(t)"=>"")
    return ret
end


function _assign_vars(vars, sym_matrix, val_matrix)
    vm = collect(Iterators.flatten(val_matrix))
    for i in eachindex(sym_matrix)
        vars[string(sym_matrix[i])] = vm[i]
    end
end


function crn_dual_binary_scalar_mult(vars, subS, subP, tspan)
    ss = species(rn_dual_binary_scalar_mult)
    varkeys = [_convert_species2var(sp) for sp in ss]
    varkeys = [replace(k, "S"=>subS) for k in varkeys]
    varkeys = [replace(k, "P"=>subP) for k in varkeys]

    # "!" ensures that the adjoint/weight gradient variables are present in vars
    uvalues = [get!(vars, k, 0.0) for k in varkeys]

    u = Pair.(ss, uvalues)
    p = []
    
    sol = simulate_reaction_network(rn_dual_binary_scalar_mult, u, p, tspan=tspan)
    
    # Assign the values to products
    for i in eachindex(ss)
        sp = string(ss[i])
        if startswith(sp, "P")  # It's a product species
            varkey = varkeys[i]
            vars[varkey] = sol[end][i]
        end
    end
end


function crn_backpropagation_step(vars, tspan; bias=0.01, reltol=1e-8, abstol=1e-8, D=2, default=0.0)
    z = _form_vector(vars, "Z")  # (D, 1) matrix
    e = _form_vector(vars, "E")
    w = _form_vector(vars, "W")

    # # Assigns the weight gradients with symbol "M" 
    # crn_dual_binary_scalar_mult(vars, "Z", "M", (0.0, 40.0)) # CHECK

    # # Assigns the adjoint variables 
    # crn_dual_binary_scalar_mult(vars, "W", "A", (0.0, 40.0)) # CHECK

    # Calculate the adjoint

    _print_vars(vars, "O", title="outputs")
    _print_vars(vars, "E", title="Errors")
    _print_vars(vars, "A", title="Adjoints..whytf are they so bad.")
    ss = species(rn_dual_backprop)

    # Initializes the gradients 
    values = []
    for sp in ss 
        push!(values, get!(vars, _convert_species2var(sp), default))
    end
    u = Pair.(ss, values)
    p = []
    
    sol = simulate_reaction_network(rn_dual_backprop, u, p, tspan=tspan, reltol=reltol, abstol=abstol) # CHECK
    return sol
end


function crn_param_update(vars, eta, tspan)
    
    ss = species(rn_param_update)
    
    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)
    
    k1 = eta/(1 + eta)
    k2 = 1/(1 + eta)
    p = [k1 k2]
    
    sol = simulate_reaction_network(rn_param_update, u, p, tspan=tspan)
    return sol
    
end


function crn_final_layer_update(vars, eta, tspan)
    ss = species(rn_final_layer_update)

    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)

    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1, k2]
    
    sol = simulate_reaction_network(rn_final_layer_update, u, p, tspan=tspan)
    return sol
end


function _dualsymbol2value(sym, mat)
    # Strip the symbol prefix
    sym = sym[2:end]
    suffix = sym[end]  # p or m 
    index_str = sym[1:end-1]  # index string
    indices = [c-'0' for c in index_str]  # indices resulting
    val = mat[indices...]
    if suffix == 'p'
        val = max(0, val)
    end
    if suffix == 'm'
        val = max(0, -val)
    end
    return val 
end


function _convert_dualsymbol2index(sym)
    sym = sym[2:end]
    suffix = sym[end]  # p or m 
    index_str = sym[1:end-1]  # index string
    indices = [c - '0' for c in index_str]  # indices resulting
    return indices
end


function dissipate_and_annihilate(vars, tspan)
    ss = species(rn_annihilation_reactions)
    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)
    p = []

    sol = simulate_reaction_network(rn_annihilation_reactions, u, p, tspan=tspan)
    return sol
    
end


function calculate_accuracy(dataset, varscopy; tspan=(0.0, 1.0), dims=3)
    acc = 0
    epoch_loss = 0.0
    for i in 1:length(dataset)
        x, y = get_one(dataset, i)
        x = augment(x, dims-length(x))

        Zspecies = [k for k in keys(varscopy) if startswith(k, "Z")]
        for zsp in Zspecies 
            varscopy[zsp] = _dualsymbol2value(zsp, x)
        end
        
        yvec = [y 1-y] # threshold could be 0.5 now.
        crn_node_forward(varscopy, (0.0, 1.0), yvec, D=dims)
        err = [varscopy["Ep"] varscopy["Em"]]
        epoch_loss += 0.5*(err[1]-err[2])^2
        output = 0.0
        if varscopy["Op"] > varscopy["Om"]
            output = 1.0
        end
        if output == y
            acc += 1
        end
    end
    _print_vars(varscopy, "Z", title="Inside calculate_accuracy")
    epoch_loss /= length(dataset)
    return epoch_loss
end


function plot_augmented_state(varscopy, dataset; tspan=(0.0, 1.0), dims=3)
    aug_x = []
    reg_x = []

    for i in eachindex(dataset)
        x, y = get_one(dataset, i)
        
        x = augment(x, dims-length(x))
        elem = []
        append!(elem, x)
        push!(elem, y)
        push!(reg_x, elem)
        
        Zspecies = [k for k in keys(varscopy) if startswith(k, "Z")]
        for zsp in Zspecies 
            varscopy[zsp] = _dualsymbol2value(zsp, x)
        end
        
        yvec = [y 1-y]
        crn_node_forward(varscopy, (0.0, 1.0), yvec, D=dims)
        push!(
            aug_x, [
                varscopy["Z1p"]-varscopy["Z1m"],
                varscopy["Z2p"]-varscopy["Z2m"],
                varscopy["Z3p"]-varscopy["Z3m"],
                yvec[1]-yvec[2]
            ]
        )
    end
    plt_state1 = scatter3d(getindex.(reg_x, 1), getindex.(reg_x, 2), getindex.(reg_x, 3), group=getindex.(reg_x, 4))
    plt_state2 = scatter3d(getindex.(aug_x, 1), getindex.(aug_x, 2), getindex.(aug_x, 3), group=getindex.(aug_x, 4))
    png(plt_state1, "before_aug.png")
    png(plt_state2, "after_aug.png")
end


function just_round_vars(vars, digits)
    for k in keys(vars)
        vars[k] = round(vars[k], digits=PRECISION_DIGITS)
    end
end

# train = create_annular_rings_dataset(100, 1.0)
# val = create_annular_rings_dataset(50, 1.0)
train = create_linearly_separable_dataset(100, linear, threshold=0.0)
val = create_linearly_separable_dataset(50, linear, threshold=0.0)


# Tracking 
DIMS = 2
TRACK_EVERY = 10
crn_epoch_losses = []
crn_epoch_train_accs = []
crn_epoch_val_accs = []
EPOCHS = 5
params_orig = create_node_params(DIMS, t0=0.0, t1=1.0)
PRECISION_DIGITS = 4
LR = 0.01
accuracy = 0.0

# Set params correctly
params = params_orig
params = custom_struct_round(params, precision_digits=PRECISION_DIGITS)

crn_tracking = Dict()

# Get the CRNs that contain all possible species to initiate vars
comprehensive_crns = [rn_dual_node_fwd, rn_dual_backprop, rn_final_layer_update, rn_param_update]

# Initiate the vars dict 
vars = Dict()
default_value = 0.0

# Initialize the vars variables
for crn in comprehensive_crns
    crn_species = species(crn)
    for s in crn_species
        get!(vars, _convert_species2var(s), default_value)
    end
end

# Assign the values to the `P` species
Pspecies = [k for k in keys(vars) if startswith(k, "P")]
for psp in Pspecies
    indices = _convert_dualsymbol2index(psp)
    @assert length(indices) == 2  # indices must be of length 2
    # @show psp, indices, params, params[(indices[1]-1)*DIMS+indices[2]]
    # The first DIMS^2 elements in the `params` array
    if endswith(psp, "p")
        vars[psp] = max(0, params[(indices[1]-1)*DIMS + indices[2]])
    end
    if endswith(psp, "m")
        vars[psp] = max(0, -params[(indices[1]-1)*DIMS + indices[2]])
    end
end

# Assign the values of the `W` variables`
Wspecies = [k for k in keys(vars) if startswith(k, "W")]
offset = DIMS^2 + 2  # 2 for tspan
for wsp in Wspecies
    indices = _convert_dualsymbol2index(wsp)
    @assert length(indices) == 1
    if endswith(wsp, "p")
        vars[wsp] = max(0, params[offset + indices[1]])
    elseif endswith(wsp, "m")
        vars[wsp] = max(0, -params[offset + indices[1]])
    end
end    

crn_epoch_val_losses = []
for epoch in 1:EPOCHS  
    epoch_loss = 0.0
    
    for i in 1:length(train)
         
        # Maybe imposing precision can help.
        just_round_vars(vars, PRECISION_DIGITS)

        x, y = get_one(train, i)
        x = augment(x, DIMS-length(x))
        println("================================")
        @show epoch, i, x, y

        Zspecies = [k for k in keys(vars) if startswith(k, "Z")]
        for zsp in Zspecies 
            vars[zsp] = _dualsymbol2value(zsp, x)
        end
        
        just_round_vars(vars, PRECISION_DIGITS)

        yvec = [y 1-y] # threshold could be 0.5 now.
        sol = crn_node_forward(vars, (0.0, 1.0), yvec, D=DIMS)

        species_dual_node_fwd = species(rn_dual_node_fwd)
        for i in eachindex(species_dual_node_fwd)
            vars[_convert_species2var(species_dual_node_fwd[i])] = sol[end][i]
        end
        just_round_vars(vars, PRECISION_DIGITS)

        yhat = dot(vars, "Z", "W")
        just_round_vars(vars, PRECISION_DIGITS)
        vars["Op"] = yhat[1]
        vars["Om"] = yhat[2]
        vars["Yp"] = yvec[1]
        vars["Ym"] = yvec[2]
        just_round_vars(vars, PRECISION_DIGITS)
        err = subtract(yhat[1], yhat[2], yvec[1], yvec[2])
        @show err
        @assert err[1]-err[2] <= 30
        vars["Ep"] = err[1]
        vars["Em"] = err[2]
        just_round_vars(vars, PRECISION_DIGITS)
        z = _form_vector(vars, "Z")  # (D, 1) matrix
        e = _form_vector(vars, "E")
        @show e
        w = _form_vector(vars, "W")
        wgradsymbolmat = _create_symbol_matrix("M", (DIMS, 1))
        wgrad = vcat([mult(e, z[i, :]) for i in 1:DIMS])
        @show wgrad
        @show wgradsymbolmat
        _assign_vars(vars, wgradsymbolmat, wgrad)
        just_round_vars(vars, PRECISION_DIGITS)
        a = reshape(vcat([mult(e, w[i, :]) for i in 1:DIMS]), size(wgrad))
        # Add the adjoint variables to var 
        asymbolmat = _create_symbol_matrix("A", (DIMS, 1))
        _assign_vars(vars, asymbolmat, a)
        just_round_vars(vars, PRECISION_DIGITS)
        # Loss add 
        epoch_loss += 0.5*(err[1] - err[2])^2 # err simulates (yhat - y) and in dual-rail notation

        #####  BACKPROPAGATION 
        sol = crn_backpropagation_step(vars, (0.0, 1.0), D=DIMS)
        # Update the gradient values
        Gspecies = [_convert_species2var(sp) for sp in species(rn_dual_backprop) if startswith(string(sp), "G")]
        for i in eachindex(Gspecies)
            vars[Gspecies[i]] = sol[end][i]
        end
        just_round_vars(vars, PRECISION_DIGITS)
        ##### Parameter update 
        sol = crn_final_layer_update(vars, LR, (0.0, 1.0))
        species_final_layer_update = species(rn_final_layer_update)
        Wspecies = [_convert_species2var(sp) for sp in species_final_layer_update if startswith(string(sp), "W")]
        for wsp in Wspecies
            vars[wsp] = sol[end][get_index_of(wsp, species_final_layer_update)]
        end
        just_round_vars(vars, PRECISION_DIGITS)
        sol = crn_param_update(vars, LR, (0.0, 1.0))
        species_param_update = species(rn_param_update)
        Pspecies = [_convert_species2var(sp) for sp in species_param_update if startswith(string(sp), "P")]
        # Update the parameter values
        for psp in Pspecies
            vars[psp] = sol[end][get_index_of(psp, species_param_update)]
        end
        just_round_vars(vars, PRECISION_DIGITS)
        _print_vars(vars, "Z", title="State after.")
        _print_vars(vars, "P", title="Parameters")
        _print_vars(vars, "W", title="Weights")
        _print_vars(vars, "M", title="Weight Gradients")

        # Zero the gradients
        for gsp in Gspecies
            vars[gsp] = 0.0
        end
        Mspecies = [k for k in keys(vars) if startswith(k, "M")]
        for msp in Mspecies
            vars[msp] = 0.0
        end
        for psp in Pspecies
            if endswith(psp, "p")
                psm = replace(psp, "p"=>"m")
                tmp = vars[psp] - vars[psm]
                vars[psp] = max(0.0, tmp)
                vars[psm] = max(0.0, -tmp)
            end
        end
        for wsp in Wspecies
            if endswith(wsp, "p")
                wsm = replace(wsp, "p"=>"m")
                tmp = vars[wsp] - vars[wsm]
                vars[wsp] = max(0.0, tmp)
                vars[wsm] = max(0.0, -tmp)
            end
        end
        # sol = dissipate_and_annihilate(vars, (0.0, 1.0))
        # species_d_and_a = species(rn_annihilation_reactions)
        # for i in eachindex(species_d_and_a)
        #     sym = _convert_species2var(species_d_and_a[i])
        #     vars[sym] = sol[end][i]
        # end
    end
    epoch_loss /= length(train)
    push!(crn_epoch_losses, epoch_loss)

    plt = plot(crn_epoch_losses)
    png(plt, "crn_losses.png")
    
end