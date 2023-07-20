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
include("reactions.jl")


function create_node_params(dims; t0=0.0, t1=1.0, precision=10)
    theta = randn(dims^2)
    for i in eachindex(theta)
        theta[i] = round(theta[i], digits=precision)
    end
    w = randn(dims)
    for i in eachindex(w)
        w[i] = round(w[i], digits=precision)
    end
    params = []
    append!(params, theta)
    push!(params, t0)
    push!(params, t1)
    append!(params, w)
    return params
end

# Adds `k` zeroes to the end of the column matrix `u`
function augment(x, k=1) # Verified!
    ret = copy(x)
    for i in 1:k
        ret = vcat(ret, 0.0)
    end
    return ret
end

# Reproducing the simple forward and backprop steps for hidden state.
function f(z, theta)
    """
    Args:
        u: hidden state
        theta: resnet parameters of size [length(u), length(u)]
    """
    fmat = theta*z
    @assert length(fmat)==length(z)
    return fmat
end


function forward!(du, u, theta, t)
    func = f(u, theta)
    for i in eachindex(func)
        du[i] = func[i]
    end
end


# Calculates the final hidden state of the neural ode
function forward_node(u0, theta, tspan)
    
    prob = ODEProblem(forward!, u0, tspan, theta)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end


# Final feedforward layer similar to a perceptron
function forward_ffnet(z, w)
    yhat = dot(w, z) # Verified!
    return yhat
end


function forward_step(u0, theta, w, tspan)
    # Output from the neural ode
    node_out = forward_node(u0, theta, tspan)
    # Extracting hidden state
    z = node_out.u[end][1:length(u0)]
    yhat = forward_ffnet(z, w)
    return (z, yhat)
end


function aug_dynamics!(du, u, theta, t)
    dims = Int32(length(theta)÷sqrt(length(theta)))
    
    # Time dynamics for the hidden state
    offset = 0
    z = u[1:dims]
    func = f(z, theta)
    
    @assert length(func) == dims
    
    for i in 1:dims
        du[offset+i] = func[i]
    end
    
    offset += dims
    
    
    # Time dynamics for the adjoint
    a = u[dims+1:2*dims]
    a = reshape(a, (dims, 1))
    
    # ∂f/∂z = 𝜃
    dfdz = theta 
    @assert size(theta) == (dims, dims)
    
    dadt = reshape(-transpose(a)*dfdz, dims)
    for i in 1:length(dadt)
        du[offset+i] = dadt[i]
    end
    
    offset += length(dadt)
    @assert offset == 2*dims # offset after adding dzdt and dady
   
    
    # Time dynamics for gradients
    dfdtheta = zeros(dims, dims^2)
    
    for i in 1:dims
        for j in 1:dims
            dfdtheta[i, (i-1)*dims + j] = z[j]
        end
    end
    
    dgrads = -transpose(a)*dfdtheta
    
    @assert size(dgrads) == (1, dims^2)
    
    for i in 1:length(dgrads)
        du[offset+i] = dgrads[i]
    end
    offset += length(dgrads)
    
    # Time dynamics of time(!!): Not used though sigh.
    # TODO: Might wanna change this in future if things don't work
    dfdt = zeros(dims, 1)
    tgrads = -transpose(a)*dfdt
    
    @assert length(tgrads) == 1
    
    # currently not changing time!
    du[offset+1] = 0 
    
end

function backpropagation_step(s0, theta, tspan)
    prob = ODEProblem(aug_dynamics!, s0, tspan, theta)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return sol
end

function sequester_params(p, dims)
    theta = zeros(dims, dims)
    for i in 1:dims^2
        theta[(i-1)÷dims + 1, (i-1)%dims + 1] = p[i]
    end
    t0 = p[dims^2 + 1]
    t1 = p[dims^2 + 2]
    w = p[dims^2+3:end]
    return theta, t0, t1, w
end

function training_step(x, y, p)
    """
    Args:
        x: augmented input
        y: output 
        p: parameters of the entire network
    """
    dims = length(x)
    
    theta, t0, t1, w = sequester_params(p, dims)
    tspan = (t0, t1)
    @assert length(w) == dims 
    
    # Forward & Hidden state calculation
    println("ODE | w at t=0 | ", w)
    z, yhat = forward_step(x, theta, w, tspan)
    z = reshape(z, (dims, 1)) # Make z a row-vector
    println("ODE | z at t=T | ", z)
    # Loss
    loss = 0.5*(yhat-y)^2
    
    # Adjoint calculation
    a = (yhat-y)*w
    a = reshape(a, (dims, 1))
    println("ODE | yhat at t=T | ", yhat)
    println("ODE | Adjoint at t=T | ", a)
    
    # Initial theta gradients
    gtheta = zeros(dims^2, 1)
    println("ODE | Gradients at t=T | ", gtheta)
    
    # Initial time gradients 
    func = f(z, theta)

    dldt1 = -transpose(a)*f(z, theta)
    dldt1 = convert(Array{Float64}, dldt1)
    
    # Initial state for the reverse time ODE
    s0 = vcat(z, a, gtheta, dldt1)
    rtspan = reverse(tspan)
    
    backward = backpropagation_step(s0, theta, rtspan)
    println("ODE | Adjoint at t=0 | ", backward.u[end][dims+1:2*dims])
    
    gradients = backward.u[end][2*dims+1:end]
    gradients = reshape(gradients, size(gradients)[1])
    println("ODE | Gradients at t=0 |  ", backward.u[end][2*dims+1:2*dims + dims^2])
    
    # Note that gradients[end] already contains gradient for t0
    append!(gradients, dldt1) # gradient for t1
    
    # Gradients wrt w
    wgrads = (yhat-y)*z
    for i in eachindex(wgrads)
        push!(gradients, wgrads[i])
    end
    println("ODE | Final layer gradients | ", wgrads)
    return z, yhat, loss, gradients
end


function predict(x, p)
    theta, t0, t1, w = sequester_params(p, length(x))
    tspan = (t0, t1)
    z, yhat = forward_step(x, theta, w, tspan) #TODO: Check tspan
    return yhat
end


function get_index_of(prefix, vec)
    ret = []
    for i in eachindex(vec)
        vec_i = vec[i]
        if startswith("$vec_i", prefix)
            push!(ret, i)
        end
    end
    
    @assert length(ret) == 1 "Prefix: $prefix, vec: $vec"
    return ret[1]
end

function get_species_array(rn)
    ret = []
    for s in species(rn)
        push!(ret, replace(string(s), "(t)" => ""))
    end
    return ret
end


# simulates multiplication of two scalars
function mult(a, b; max_val=100.0)
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


function _convert_dot2var(s, subA, subB)
     ret = s 
     ret = replace(ret, "(t)" => "")
     ret = replace(ret, "A" => subA)
     ret = replace(ret, "B" => subB)
     return ret 
end

function subtract(ap, am, bp, bm; max_val=100.0, default=0.0)
    u = [
        :Ap => ap, :Am => am, :Bp => bp, :Bm => bm, :Yp => default, :Ym => default
    ]
    p = []
    sol = simulate_reaction_network(rn_dual_subtract, u, p, tspan=(0.0, max_val))
    
    ss = species(rn_dual_subtract)
    yp = sol[end][get_index_of("Yp", ss)]
    ym = sol[end][get_index_of("Ym", ss)]
    y = [yp ym] 
    return y
end


function dot2(vars, subA, Aspecies, subB, Bspecies; max_val=100.00, reltol=1e-8, abstol=1e-8, default=0.0)
    dot2ss = species(rn_dual_dot2)

    # Initial concentration values
    varkeys = [_convert_dot2var(string(sp), subA, subB)  for sp in dot2ss]
    uvalues = [get(vars, k, default) for k in varkeys]
    @show varkeys
    @show uvalues
    u = Pair.(dot2ss, uvalues)

    # solve the ODE
    p = []
    sol = simulate_reaction_network(rn_dual_dot2, u, p, tspan=(0.0, max_val), reltol=reltol, abstol=abstol)

    # Collect the outputs
    yp = sol[end][get_index_of("Yp", dot2ss)]
    ym = sol[end][get_index_of("Ym", dot2ss)]
    y = [yp ym]
    return y 
end


function convert_to_dual_rail(x)
    return (max(0, x), abs(min(0, x)))
end


function crn_node_forward(vars, tspan, y; reltol=1e-8, abstol=1e-8, precision=4, D=2)
    # println("============= CRN FORWARD STEP =====================")
    # Get the species in the order in which the function `species` retrieves them
    sps = get_species_array(rn_dual_node_fwd)
    
    uvalues = [vars[sp] for sp in sps]
    # Assigning initial values
    u = Pair.(species(rn_dual_node_fwd), uvalues)
    p = []  # All the reactions have unit rate constant, stays empty
    
    sol = simulate_reaction_network(rn_dual_node_fwd, u, p, tspan=tspan, reltol=reltol, abstol=abstol)

    for sp in sps
        vars[sp] = sol[end][get_index_of(sp, sps)]
    end
    
    # Calculate the dot product of Z and W. We also utilize `D` here
    Zspecies = [sp for sp in keys(vars) if startswith(sp, "Z")]
    Wspecies = [sp for sp in keys(vars) if startswith(sp, "W")]
    
    
    yhat = dot2(vars, "Z", Zspecies, "W", Wspecies) 

    e = subtract(yhat[1], yhat[2], y[1], y[2])
    @show yhat
    @show y
    @show e
    for wsp in Wspecies 
        @show vars[wsp]
    end

    # @assert e[1] <= 100
    # @assert e[2] <= 100
    vars["Op"] = yhat[1]
    vars["Om"] = yhat[2]
    vars["Ep"] = e[1]
    vars["Em"] = e[2]

    # println("============= END CRN FORWARD STEP ===============")
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


function _init_variables(prefix, data)
    # Note: Data here is not provided in dual_rail
    dims = size(data)
    @assert length(dims) == 2
    suffixes = []
    rows = dims[1]
    cols = dims[2]
    values = []
    for row in 1:rows
        for col in 1:cols
            p = max(0, data[row, col])
            m = max(0, -data[row, col])
            push!(suffixes, string(prefix, string(row), string(col), "p"))
            push!(values, p)
            push!(suffixes, string(prefix, string(row), string(col), "m"))
            push!(values, m)
        end
    end
    return Dict(Pair.(suffixes, values))
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


function crn_backpropagation_step(vars, tspan; bias=0.01, reltol=1e-8, abstol=1e-8, D=2, default=0.0)
    
    z = _form_vector(vars, "Z")  # (D, 1) matrix
    e = _form_vector(vars, "E")
    w = _form_vector(vars, "W")
    @show z
    @show e
    @show w
    # Calculate output layer weight gradients
    wgradsymbolmat = _create_symbol_matrix("M", (D, 1))
    wgrad = vcat([mult(e, z[i, :]) for i in 1:D])
    @show wgrad
    @show wgradsymbolmat
    _assign_vars(vars, wgradsymbolmat, wgrad)

    # Calculate the adjoint 
    a = reshape(vcat([mult(e, w[i, :]) for i in 1:D]), size(wgrad))
    
    # Add the adjoint variables to var 
    asymbolmat = _create_symbol_matrix("A", (D, 1))
    _assign_vars(vars, asymbolmat, a)

    ss = species(rn_dual_backprop)

    # Initializes the gradients 
    values = []
    for sp in ss 
        push!(values, get!(vars, _convert_species2var(sp), default))
    end
    u = Pair.(ss, values)
    p = []
    
    sol = simulate_reaction_network(rn_dual_backprop, u, p, tspan=tspan, reltol=reltol, abstol=abstol) # CHECK
    ss = species(rn_dual_backprop)


end


function crn_weight_update(vars, eta, tspan)
    
    ss = species(rn_weight_update)
    
    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)
    
    k1 = eta/(1 + eta)
    k2 = 1/(1 + eta)
    p = [k1, k2]
    
    sol = simulate_reaction_network(rn_weight_update, u, p, tspan=tspan)

    Pspecies = [_convert_species2var(sp) for sp in ss if startswith(string(sp), "P")]


    # Update the parameter values
    for psp in Pspecies
        vars[psp] = sol[end][get_index_of(psp, ss)]
    end
end


function crn_final_layer_update(vars, eta, tspan)
    ss = species(rn_final_layer_update)

    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)

    k1 = eta / (1 + eta)
    k2 = 1 / (1 + eta)
    p = [k1, k2]
    
    sol = simulate_reaction_network(rn_final_layer_update, u, p, tspan=tspan)
    
    Wspecies = [_convert_species2var(sp) for sp in ss if startswith(string(sp), "W")]
    for wsp in Wspecies
        vars[wsp] = sol[end][get_index_of(wsp, ss)]
    end
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
    indices = [c-'0' for c in index_str]  # indices resulting
    @assert length(indices) == 2
    return indices
end


function dissipate_and_annihilate(vars, tspan)
    ss = species(rn_annihilation_reactions)
    uvalues = [vars[_convert_species2var(sp)] for sp in ss]
    u = Pair.(ss, uvalues)
    p = []

    sol = simulate_reaction_network(rn_annihilation_reactions, u, p, tspan=tspan)

    for i in eachindex(ss)
        sym = _convert_species2var(ss[i])
        vars[sym] = sol[end][i]
    end
end


################################################################################################################################################################################################################################################################################################################

# Dataset 

train = create_linearly_separable_dataset!(100, linear, threshold=0.0)
val = create_linearly_separable_dataset!(20, linear, threshold=0.0)
# train = create_annular_rings_dataset!(100, 1.0)
# val = create_annular_rings_dataset!(20, 1.0)



# Tracking 
DIMS = 2
TRACK_EVERY = 10
crn_epoch_losses = []
EPOCHS = 10
params_orig = create_node_params(DIMS, t0=0.0, t1=1.0)
PRECISION_DIGITS = 4
LR = 0.001
accuracy = 0.0

# Set params correctly
params = params_orig
params = custom_struct_round(params, precision_digits=PRECISION_DIGITS)

crn_tracking = Dict()

# Get the CRNs that contain all possible species to initiate vars
comprehensive_crns = [rn_dual_node_fwd, rn_dual_backprop, rn_weight_update, rn_final_layer_update]

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

# The first DIMS^2 elements in the `params` array
vars[psp] = params[indices[1]*DIMS + indices[2]]
end


for epoch in 1:EPOCHS  
epoch_loss = 0.0
for i in 1:length(train)
         
        x, y = get_one(train, i)
        x = augment(x, DIMS-length(x))

        println("epoch: ", epoch, " i: ", i, " x: ", x, "\n===========")

        Zspecies = [k for k in keys(vars) if startswith(k, "Z")]
        for zsp in Zspecies 
            @show zsp, x 
            vars[zsp] = _dualsymbol2value(zsp, x)
            @show vars[zsp]
        end
        
        
        crn_node_forward(vars, (0.0, 1.0), [y 1.0-y])

        err = [vars["Ep"] vars["Em"]]
        
        # Loss add 
        epoch_loss += 0.5*(err[1] - err[2])^2 # err simulates (yhat - y) and in dual-rail notation
        crn_backpropagation_step(vars, (0.0, 1.0))
        crn_weight_update(vars, LR, (0.0, 10.0))
        crn_final_layer_update(vars, LR, (0.0, 10.0))

        # Tracking
        Pspecies = [k for k in keys(vars) if startswith(k, "P")]
        for psp in Pspecies
            get!(crn_tracking, psp, [])
            push!(crn_tracking[psp], vars[psp])
        end
        Gspecies = [k for k in keys(vars) if startswith(k, "G")]
        for gsp in Gspecies
            get!(crn_tracking, gsp, [])
            push!(crn_tracking[gsp], vars[gsp])
        end
        
        # Zero the gradients
        Mspecies = [k for k in keys(vars) if startswith(k, "M")]
        for msp in Mspecies
            vars[msp] = 0.0
        end
        for gsp in Gspecies
            vars[gsp] = 0.0
        end
        dissipate_and_annihilate(vars, (0.0, 10.0))
        
    end
    epoch_loss /= length(train)
    push!(crn_epoch_losses, epoch_loss)
    plt = plot(crn_epoch_losses)
    png(plt, "crn_losses.png")
    
end

acc = []
for i in eachindex(val)
    x, y = get_one(val, i)
    x = augment(x, DIMS-length(x))
    Zspecies = [k for k in keys(vars) if startswith(k, "Z")]
    for zsp in Zspecies 
        vars[zsp] = _dualsymbol2value(zsp, x)
    end
    
    crn_node_forward(vars, (0.0, 1.0), [y 1.0-y])
    output = vars["Op"] - vars["Om"]
    exp = 0.0
    if output > 0.0
        exp = 1.0
    end
    if exp == y
        push!(acc, 1)
    end
    println(exp, "  ", y, " ", output)
end
println("Acc: ", length(acc)/length(val))


