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

    D = 5
    NUM_CLASSES = 3
    
    z = ode.Matrix2D(symbol='z', dims=[D, 1])
    p = ode.Matrix2D(symbol='p', dims=[D, D])
    x = ode.Matrix2D(symbol='x', dims=[D, 1])
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    grads = ode.Matrix2D(symbol='g', dims=[D, D])._reshape(dims=[1, D**2])
    h = ode.Matrix2D(symbol='h', dims=[D, 1])
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
    # lcs = utils.print_crn(fwd_zz_crn)
    for i in range(D):
        print(f"100.0, Z{i}p + Z{i}m --> 0")
        print(f"1.0, Z{i}p + Z{i}p --> Z{i}p")
        print(f"1.0, Z{i}m + Z{i}m --> Z{i}m")
    print("end")

# ------------------------------------------
    ## [yhat1 yhat2 yhat3 =  wT*z]
    print("""rn_yhat_calculate = @reaction_network rn_yhat_calculate begin""")
    w = ode.Matrix2D(symbol='w', dims=[D, NUM_CLASSES])
    o = ode.Matrix2D(symbol='o', dims=[NUM_CLASSES, 1])
    wT = ode.transpose_matrix(w)
    fwd_zw_ode = ode.ODESystem(lhs=o, rhs=[wT, z])
    fwd_zw_crn = fwd_zw_ode.dual_rail_crn()
    lcs = utils.print_crn(fwd_zw_crn)
    for i in range(NUM_CLASSES):
        print(f"1.0, O{i}p --> 0")
        print(f"1.0, O{i}m --> 0")
    print("end")

    ################## Backprop ###############

    # a = dL/dz = W e 
    print("# Calculate the adjoint")
    print("rn_adjoint_calculate = @reaction_network rn_adjoint_calculate begin")
    a = ode.Matrix2D(symbol='a', dims=[D, 1])
    w = ode.Matrix2D(symbol='w', dims=[D, NUM_CLASSES])
    e = ode.Matrix2D(symbol='E', dims=[NUM_CLASSES, 1])
    calc_adj_ode = ode.ODESystem(lhs=a, rhs=[w, e])
    calc_adj_crn = calc_adj_ode.dual_rail_crn()
    lcs = utils.print_crn(calc_adj_crn)
    for i in range(D):
        print(f"1.0, A{i}p --> 0")
        print(f"1.0, A{i}m --> 0")
    print("end")


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
    # lcs = utils.print_crn(bwd_zz_crn)
    for i in range(D): 
        print(f"100.0, Z{i}p + Z{i}m --> 0")
        print(f"1.0, Z{i}p + Z{i}p --> Z{i}p + Z{i}p + Z{i}p")
        print(f"1.0, Z{i}m + Z{i}m --> Z{i}m + Z{i}m + Z{i}m")
    
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
    print("end")
    
    # wgrads = Z eT
    print("# Calculate wgrads")
    e = ode.Matrix2D(symbol='E', dims=[NUM_CLASSES, 1])
    eT = ode.transpose_matrix(e)
    wgrads = ode.Matrix2D(symbol='m', dims=[D, NUM_CLASSES])
    bwd_wgrads_ode = ode.ODESystem(lhs=wgrads, rhs=[z, eT], parity=1)
    bwd_wgrads_crn = bwd_wgrads_ode.dual_rail_crn()
    print("""rn_wgrads_calculate = @reaction_network rn_wgrads_calculate begin""")
    lcs = utils.print_crn(bwd_wgrads_crn)
    for i in range(D):
        for j in range(NUM_CLASSES):
            print(f"1.0, M{i}{j}p --> 0")
            print(f"1.0, M{i}{j}m --> 0")
    print("""end""")


    # rn dual dot
    sa = ode.Matrix2D(symbol='a', dims=[D, 1])
    sb = ode.Matrix2D(symbol='b', dims=[D, 1])
    py = ode.Matrix2D(symbol='Y', dims=[D, 1])
    sabhdmd = sa._hadamard(b)
    dual_dot_ode = ode.ODESystem(lhs=py, rhs=[sabhdmd], parity=1)
    dual_dot_crn = dual_dot_ode.dual_rail_crn()
    print("# dy/dt = ab. Add Yp --> 0 and Ym --> 0")
    print("""rn_dual_dot = @reaction_network rn_dual_dot begin""")
    lcs = utils.print_crn(dual_dot_crn)
    print("""end""")


    print("""# rn_param_update""")
    print("""rn_param_update = @reaction_network rn_param_update begin""")
    for i in range(D):
        for j in range(D):
            print(f"""k1, G{i}{j}p --> P{i}{j}m""")
            print(f"""k1, G{i}{j}m --> P{i}{j}p""")
            print(f"""k2, G{i}{j}p --> 0""")
            print(f"""k2, G{i}{j}m --> 0""")
    for i in range(D):
        print(f"""k1, V{i}p --> B{i}m""")
        print(f"""k1, V{i}m --> B{i}p""")
        print(f"""k2, V{i}p --> 0""")
        print(f"""k2, V{i}m --> 0""")
    print("""end""")

    print("""# rn_final_layer_update""")
    print("""rn_final_layer_update = @reaction_network rn_final_layer_update begin""")
    for i in range(D):
        for j in range(NUM_CLASSES):
            print(f"""k1, M{i}{j}p --> W{i}{j}m""")
            print(f"""k1, M{i}{j}m --> W{i}{j}p""")
            print(f"""k2, M{i}{j}p --> 0""")
            print(f"""k2, M{i}{j}m --> 0""")
    print("""end""")

    print("# Create error species")
    print("rn_create_error_species = @reaction_network rn_create_error_species begin")
    for i in range(NUM_CLASSES):
        print(f"""10.0, O{i}p --> E{i}p
10.0, Y{i}p --> E{i}m
10.0, O{i}m --> E{i}m
10.0, Y{i}m --> E{i}p
100.0, E{i}p + E{i}m --> 0""")
    print("end")