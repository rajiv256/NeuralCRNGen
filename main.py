import src.ode as ode
import utils
import numpy as np

if __name__ == '__main__':
    D = 2
    z = ode.Variable(symbol='z', dimensions=[D])
    P = ode.Variable(symbol='P', dimensions=[D, D])
    a = ode.Variable(symbol='a', dimensions=[D])
    g = ode.Variable(symbol='g', dimensions=[D, D])
    dfdz = ode.Variable(symbol='dfdz', dimensions=[D, D],
                        data=[['p11', 'p12'], ['p21', 'p22']])
    dfdtheta = ode.Variable(symbol='dfdtheta', dimensions=[D, D**2],
                             data=ode.create_dfdtheta(D, prefix='z'))

    concentrations = []
    # Node fwd
    fwd_z_ode = ode.ODESystem(lhs=z, rhs=[P, z], parity=1)
    fwd_z_crn = fwd_z_ode.dual_rail_crn()
    concentrations += utils.print_crn(fwd_z_crn, title="CRN for Z Forward")

    # Z backprop
    bwd_z_ode = ode.ODESystem(lhs=z, rhs=[P, z], parity=-1)
    bwd_z_crn = bwd_z_ode.dual_rail_crn()
    concentrations += utils.print_crn(bwd_z_crn, title="CRN for Z Backprop")

    # Adjoint Backprop
    aT = ode.transpose_variable(a)  # Use suffix T only for transpose
    # TODO: Requires parity check
    bwd_adj_ode = ode.ODESystem(lhs=aT, rhs=[aT, dfdz], parity=1)
    bwd_adj_crn = bwd_adj_ode.dual_rail_crn()
    concentrations += utils.print_crn(bwd_adj_crn, title="CRN for Adjoint "
                                                      "Backprop")

    # Gradient backprop
    gT = ode.transpose_variable(g)
    gT.data = np.array(gT.data).flatten().reshape(1, -1).tolist()
    bwd_g_ode = ode.ODESystem(lhs=gT, rhs=[aT, dfdtheta], parity=1)
    bwd_g_crn = bwd_g_ode.dual_rail_crn()
    concentrations += utils.print_crn(bwd_g_crn, title="CRN for Gradient "
                                                     "Backprop")

    # Parameter update


    concentrations = set(concentrations)
    print(concentrations)