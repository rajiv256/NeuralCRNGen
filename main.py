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

    D = 2  # Input
    
    z = ode.Matrix2D(symbol='z', dims=[D, 1])
    p = ode.Matrix2D(symbol='p', dims=[D, 1])
    x = ode.Matrix2D(symbol='x', dims=[D, 1])
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    grads = ode.Matrix2D(symbol='g', dims=[D, 1])
    

    # ReLU node forward
    print("rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin")
    
    ## dz/dt = p Hadamard z (It's okay because Bi is not used anywhere else)
    pxhdmd = p._hadamard(x)
    fwd_px_ode = ode.ODESystem(lhs=z, rhs=[pxhdmd])
    print("# dz_i/dt = p_ix_i")
    fwd_px_crn = fwd_px_ode.dual_rail_crn()
    lcs = utils.print_crn(fwd_px_crn)
    print("end")
   
    ################## Backprop
    print("rn_dual_backprop = @reaction_network rn_dual_backprop begin")
    bwd_px_ode = ode.ODESystem(lhs=z, rhs=[pxhdmd], parity=-1)
    bwd_px_crn = bwd_px_ode.dual_rail_crn()
    print("# dz/dt = -p_i x_i")
    lcs = utils.print_crn(bwd_px_crn)

    # # A backward
    # aphdmd = a._hadamard(p)
    # bwd_ap_ode = ode.ODESystem(lhs=a, rhs=[aphdmd], parity=1) # May 14, 2024 changed parity 1.
    # bwd_ap_crn = bwd_ap_ode.dual_rail_crn()
    # print("\n# da_i/dt = a_i p_i") # f--> -f
    # lcs = utils.print_crn(bwd_ap_crn)
    
    # Parameter gradients
    axhdmd = a._hadamard(x)
    bwd_pgrads_ode = ode.ODESystem(lhs=grads, rhs=[axhdmd], parity=1)
    bwd_pgrads_crn = bwd_pgrads_ode.dual_rail_crn()
    print("# dg_i/dt = a_i x_i") # f--> -f, parity = 1
    lcs = utils.print_crn(bwd_pgrads_crn)
    print("end")
    