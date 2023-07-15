from pprint import pprint
import src.ode as ode
import utils
import numpy as np
from src.algebra import Scalar

if __name__ == '__main__':
    D = 2
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
    dfdtheta = ode.Matrix2D(symbol='dfdtheta', dims=[D, D**2],
                             data=ode.create_dfdtheta(D, prefix='z'))

    concentrations = []
    # Node fwd
    fwd_z_ode = ode.ODESystem(lhs=z, rhs=[p, z], parity=1)
    fwd_z_crn = fwd_z_ode.dual_rail_crn()
    loc_concentrations = utils.print_crn(fwd_z_crn, title="CRN for Z Forward")
    concentrations += loc_concentrations
    pprint(set(loc_concentrations))

    # Z backprop
    bwd_z_ode = ode.ODESystem(lhs=z, rhs=[p, z], parity=-1)
    bwd_z_crn = bwd_z_ode.dual_rail_crn()
    loc_concentrations = utils.print_crn(bwd_z_crn, title="CRN for Z Backprop")
    concentrations += loc_concentrations
    pprint(set(loc_concentrations))

    # Adjoint Backprop
    aT = ode.transpose_matrix(a)  # Use suffix T only for transpose
    # TODO: Requires parity check
    bwd_adj_ode = ode.ODESystem(lhs=aT, rhs=[aT, dfdz], parity=1)
    bwd_adj_crn = bwd_adj_ode.dual_rail_crn()
    loc_concentrations = utils.print_crn(bwd_adj_crn, title="CRN for Adjoint "
                                                      "Backprop")
    concentrations += loc_concentrations
    pprint(set(loc_concentrations))

    # Gradient backprop
    gg = ode.Matrix2D(symbol=g.symbol, data=np.array(g.data).flatten(

    ).reshape(1, -1))
    bwd_g_ode = ode.ODESystem(lhs=gg, rhs=[aT, dfdtheta], parity=1)
    bwd_g_crn = bwd_g_ode.dual_rail_crn()
    loc_concentrations = utils.print_crn(bwd_g_crn, title="CRN for Gradient "
                                                     "Backprop")
    concentrations += loc_concentrations
    pprint(set(loc_concentrations))

    # Parameter update
    concentrations = set(concentrations)
    pprint(concentrations)
    print(concentrations)