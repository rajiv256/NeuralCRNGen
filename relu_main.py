import numpy as np
import src.ode as ode
import utils
from src.algebra import Scalar, Term
import itertools


def print_dual_dot_crn(a="a", b="b", y="y", D=2):
    reactions = []
    ascalars = [Scalar(a + str(i + 1)) for i in range(D)]
    bscalars = [Scalar(b + str(i + 1)) for i in range(D)]
    yscalars = [Scalar(y)]
    amat = ode.Matrix2D(dims=[1, D], data=ascalars)
    bmat = ode.Matrix2D(dims=[D, 1], data=bscalars)
    ymat = ode.Matrix2D(dims=[1, 1], data=yscalars)

    dotode = ode.ODESystem(lhs=ymat, rhs=[amat, bmat], parity=1)
    dotcrn = dotode.dual_rail_crn()
    print("rn_dual_mult = @reaction_network rn_dual_mult begin")
    utils.print_crn(dotcrn, title='NA')
    print("end\n\n")


def print_gradient_update_crn(
        k1="k1", k2="k2", P="P", G="G", dims=[], title=''
):
    assert 2 >= len(dims) > 0
    if len(dims) == 1:
        d1 = dims[0]
        indices = [str(x) for x in range(1, d1 + 1)]
    if len(dims) == 2:
        d1 = dims[0]
        d2 = dims[1]
        pair_indices = list(
            itertools.product(range(1, d1 + 1), range(1, d2 + 1))
        )
        indices = [str(x) + str(y) for x, y in pair_indices]
    print(f"{title} = @reaction_network {title}  begin")
    for index in indices:
        print(f"{k1}, {G}{index}p --> {P}{index}m")
        print(f"{k1}, {G}{index}m --> {P}{index}p")

    for index in indices:
        print(f"{k2}, {G}{index}p --> 0")
        print(f"{k2}, {G}{index}m --> 0")
    print(f"end {k1} {k2}\n\n")


if __name__ == '__main__':

    D = 3
    
    z = ode.Matrix2D(symbol='z', dims=[D, 1])
    p = ode.Matrix2D(symbol='p', dims=[D, D])
    x = ode.Matrix2D(symbol='x', dims=[D, 1])
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    grads = ode.Matrix2D(symbol='g', dims=[D, D])._reshape(dims=[1, D**2])
    h = ode.Matrix2D(symbol='h', dims=[1, 1])
    b = ode.Matrix2D(symbol='b', dims=[D, 1])
    bgrads = ode.Matrix2D(symbol='v', dims=[D, 1])
    

    # ReLU node forward
    print("rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin")
    
    ## dz/dt = h
    fwd_h_ode = ode.ODESystem(lhs=z, rhs=[h], parity=1)
    fwd_h_crn = fwd_h_ode.dual_rail_crn()
    print("# dz_i/dt = h")
    lcs = utils.print_crn(fwd_h_crn)

    ## dz_i/dt = sum p_ij x_j z_i
    pzhdmd = p._hadamard(z)
    fwd_pzx_ode = ode.ODESystem(lhs=z, rhs=[pzhdmd, x], parity=1)
    fwd_pzx_crn = fwd_pzx_ode.dual_rail_crn()
    print("# dz_i/dt = p_ij x_j z_i")
    lcs = utils.print_crn(fwd_pzx_crn)

    ## dz/dt = bz (It's okay because Bi is not used anywhere else)
    bzhdmd = b._hadamard(z)
    fwd_bz_ode = ode.ODESystem(lhs=z, rhs=[bzhdmd])
    print("# dz_i/dt = b_iz_i")
    fwd_bz_crn = fwd_bz_ode.dual_rail_crn()
    lcs = utils.print_crn(fwd_bz_crn)
    
    ## dz/dt = -z^2
    zzhdmd = z._hadamard(z)
    fwd_zz_ode = ode.ODESystem(lhs=z, rhs=[zzhdmd], parity=-1) # Notice parity
    fwd_zz_crn = fwd_zz_ode.dual_rail_crn()
    print("# dz_i/dt = -z_i^2")
    lcs = utils.print_crn(fwd_zz_crn)

    ################## Backprop

    print("rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin")
    # Z backprop
    bwd_h_ode = ode.ODESystem(lhs=z, rhs=[h], parity=-1)
    bwd_h_crn = bwd_h_ode.dual_rail_crn()
    print("# dz/dt = -h")
    lcs = utils.print_crn(bwd_h_crn)

    bwd_pzx_ode = ode.ODESystem(lhs=z, rhs=[pzhdmd, x], parity=-1)
    bwd_pzx_crn = bwd_pzx_ode.dual_rail_crn()
    print("# dz/dt = -p_ij x_j z_i")
    lcs = utils.print_crn(bwd_pzx_crn)

    bwd_bz_ode = ode.ODESystem(lhs=z, rhs=[bzhdmd], parity=-1)
    bwd_bz_crn = bwd_bz_ode.dual_rail_crn()
    print("# dz/dt = -b_i z_i")
    lcs = utils.print_crn(bwd_bz_crn)

    bwd_zz_ode = ode.ODESystem(lhs=z, rhs=[zzhdmd], parity=1) # parity is 1
    bwd_zz_crn = bwd_zz_ode.dual_rail_crn()
    print("# dz/dt = z^2")
    lcs = utils.print_crn(bwd_zz_crn)


    # A backward
    aphdmd = a._hadamard(p)
    bwd_apx_ode = ode.ODESystem(lhs=a, rhs=[aphdmd, x], parity=-1)
    bwd_apx_crn = bwd_apx_ode.dual_rail_crn()
    print("\n# da_i/dt = a_i p_ij x_j") # f--> -f
    lcs = utils.print_crn(bwd_apx_crn)
    
    # Gradients
    amat = a.matrix()
    amat = amat.repeat(D) # [a1 a1.. a3 a3 a3]
    arepeat = ode.Matrix2D(symbol='A', dims=[1, D**2], data=amat.flatten().tolist())
    
    zmat = z.matrix().repeat(D)
    zrepeat = ode.Matrix2D(symbol='Z', dims=[1, D**2],
                           data=zmat.flatten().tolist())
        
    xmat = x.matrix().repeat(D).reshape([D, D]).transpose()
    xrepeat = ode.Matrix2D(symbol='X', dims=[1, D**2],
                           data=xmat.flatten().tolist())
    
    azrepeathdmd = arepeat._hadamard(zrepeat)
    azxrepeathdmd = azrepeathdmd._hadamard(xrepeat)
    
    bwd_grads_ode = ode.ODESystem(lhs=grads, rhs=[azxrepeathdmd], parity=1)
    bwd_grads_crn = bwd_grads_ode.dual_rail_crn()
    print("# dg_ij/dt = a_i z_i x_j") # f --> -f ,  parity = 1
    lcs = utils.print_crn(bwd_grads_crn) 

    # Beta gradients
    azhdmd = a._hadamard(z)
    bwd_bgrads_ode = ode.ODESystem(lhs=bgrads, rhs=[azhdmd], parity=1)
    bwd_bgrads_crn = bwd_bgrads_ode.dual_rail_crn()
    print("# dbg_i/dt = a_i z_i") # f--> -f, parity = 1
    lcs = utils.print_crn(bwd_bgrads_crn)

