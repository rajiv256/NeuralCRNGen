import numpy as np
import src.ode as ode
import utils
from src.algebra import Scalar
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
    D = 2  # TODO: INPUT
    z = ode.Matrix2D(symbol='z', dims=[D, 1])
    p = ode.Matrix2D(symbol='p', dims=[D, D])
    p.matrix()
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    g = ode.Matrix2D(symbol='g', dims=[D, D])
    gmat = g.matrix()  # This assigns g.data to a proper value
    # TODO: The value of dfdz is okay for now but changes when $f$ changes
    #  from \theta z to RelU
    dfdz = ode.Matrix2D(symbol='dfdz', dims=[D, D], data=p.data)
    dfdtheta_data = ode.create_dfdtheta(D, prefix='z')
    dfdtheta = ode.Matrix2D(
        symbol='dfdtheta', dims=[D, D ** 2],
        data=ode.create_dfdtheta(D, prefix='z')
    )

    # Node fwd
    print("rn_dual_node_fwd = @reaction_network rn_dual_node_fwd begin")
    fwd_z_ode = ode.ODESystem(lhs=z, rhs=[p, z], parity=1)
    fwd_z_crn = fwd_z_ode.dual_rail_crn()
    lcs = utils.print_crn(fwd_z_crn, title='NA')
    print("end")
    print("\n\n")

    # Z backprop
    print("rn_dual_backprop = @reaction_network rn_dual_backprop begin")
    print("## Hidden state backprop")
    bwd_z_ode = ode.ODESystem(lhs=z, rhs=[p, z], parity=-1)
    bwd_z_crn = bwd_z_ode.dual_rail_crn()
    lcs = list(set(utils.print_crn(bwd_z_crn, title="CRN for Z Backprop")))

    # Adjoint Backprop
    print("## Adjoint state backprop")
    aT = ode.transpose_matrix(a)  # Use suffix T only for transpose

    # TODO: Requires parity check
    bwd_adj_ode = ode.ODESystem(lhs=aT, rhs=[aT, dfdz], parity=1)
    bwd_adj_crn = bwd_adj_ode.dual_rail_crn()
    lcs = list(
        set(
            utils.print_crn(
                bwd_adj_crn, title="CRN for "
                                   "Adjoint "
                                   "Backprop"
            )
        )
    )

    # Gradient backprop
    print("## Gradient backprop")
    gg = ode.Matrix2D(
        symbol=g.symbol, data=np.array(g.data).flatten().reshape(1, -1)
    )
    bwd_g_ode = ode.ODESystem(lhs=gg, rhs=[aT, dfdtheta], parity=1)
    bwd_g_crn = bwd_g_ode.dual_rail_crn()
    lcs = list(set(utils.print_crn(bwd_g_crn, title='NA')))
    print("end")
    print("\n\n")

    print_dual_dot_crn()  # This is a generic CRN.

    print_gradient_update_crn(title="rn_gradient_update", dims=[D, D])
    print_gradient_update_crn(title="rn_final_layer_update",
                              P="W", G="M", dims=[D])

    #