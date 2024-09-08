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

    # trying out: f = \theta \dot x - y 
    D = 3
    z = ode.Matrix2D(symbol='z', dims=[D, 1])
    p = ode.Matrix2D(symbol='p', dims=[D, 1])
    x = ode.Matrix2D(symbol='x', dims=[D, 1])
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    grads = ode.Matrix2D(symbol='g', dims=[D, 1])
    h = ode.Matrix2D(symbol='h', dims=[1, 1])
    b = ode.Matrix2D(symbol='b', dims=[D, 1])
    bgrads = ode.Matrix2D(symbol='v', dims=[D, 1])
    

    # ReLU node forward
    print("rn_ncrn_fwd = @reaction_network rn_ncrn_fwd begin")

    ## dz/dt = h
    fwd_h_ode = ode.ODESystem(lhs=z, rhs=[h], parity=1)
    fwd_h_crn = fwd_h_ode.dual_rail_crn()
    print("# dz_i/dt = h")
    lcs = utils.print_crn(fwd_h_crn)

    ## dz_i/dt = p_i z_i
    pzhdmd = p._hadamard(x)
    fwd_pz_ode = ode.ODESystem(lhs=z, rhs=[pzhdmd], parity=1)
    fwd_pz_crn = fwd_pz_ode.dual_rail_crn()
    print("# dz_i/dt = p_i z_i")
    lcs = utils.print_crn(fwd_pz_crn)
    
    # ## dz/dt = -z^2
    # zzhdmd = z._hadamard(z)
    # fwd_zz_ode = ode.ODESystem(lhs=z, rhs=[z], parity=-1) # Notice parity
    # fwd_zz_crn = fwd_zz_ode.dual_rail_crn()
    # print("# dz_i/dt = -z^2")
    # lcs = utils.print_crn(fwd_zz_crn)
    # print("end")

    ################## Backprop #####################

    print("rn_ncrn_bwd = @reaction_network rn_ncrn_bwd begin")
    # Z backprop

    bwd_h_ode = ode.ODESystem(lhs=z, rhs=[h], parity=-1)
    bwd_h_crn = bwd_h_ode.dual_rail_crn()
    print("# dz/dt = -h")
    lcs = utils.print_crn(bwd_h_crn)


    pzhdmd = p._hadamard(x)
    bwd_pz_ode = ode.ODESystem(lhs=z, rhs=[pzhdmd], parity=-1)
    bwd_pz_crn = bwd_pz_ode.dual_rail_crn()
    print("# dz/dt = -p_i z_i")
    lcs = utils.print_crn(bwd_pz_crn)

    # zzhdmd = z._hadamard(z)
    # bwd_zz_ode = ode.ODESystem(lhs=z, rhs=[z], parity=1)
    # bwd_zz_crn = bwd_zz_ode.dual_rail_crn()
    # print("# dz/dt = z^2")
    # lcs = utils.print_crn(bwd_zz_crn)

    # A backward
    # aphdmd = a._hadamard(p)
    # bwd_a_ode = ode.ODESystem(lhs=a, rhs=[aphdmd], parity=1)
    # bwd_a_crn = bwd_a_ode.dual_rail_crn()
    # print("\n# da_i/dt = a_i p_i")
    # lcs = utils.print_crn(bwd_a_crn)

    azhdmd = a._hadamard(z)
    bwd_a_ode = ode.ODESystem(lhs=a, rhs=[azhdmd], parity=-1)
    bwd_a_crn = bwd_a_ode.dual_rail_crn()
    print("\n# da_i/dt = -a_i z_i")
    lcs = utils.print_crn(bwd_a_crn)

    # Gradients
    azhdmd = a._hadamard(z)
    bwd_grads_ode = ode.ODESystem(lhs=grads, rhs=[azhdmd], parity=1)
    bwd_grads_crn = bwd_grads_ode.dual_rail_crn()
    print("# dg_i/dt = a_i z_i") # f --> -f ,  parity = 1
    lcs = utils.print_crn(bwd_grads_crn)
    print("end")
