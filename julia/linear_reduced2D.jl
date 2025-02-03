rn_dual_node_fwd = @reaction_network rn_dual_node_fwd begin
    # dz_i/dt = p_ix_i
    1.0, P1p + X1p --> Z1p + P1p + X1p
    # 1.0, P1m + X1m --> Z1p + P1m + X1m
    # 1.0, P1p + X1m --> Z1m + P1p + X1m
    # 1.0, P1m + X1p --> Z1m + P1m + X1p
    1.0, P2p + X2p --> Z2p + P2p + X2p
    # 1.0, P2m + X2m --> Z2p + P2m + X2m
    # 1.0, P2p + X2m --> Z2m + P2p + X2m
    # 1.0, P2m + X2p --> Z2m + P2m + X2p

    1.0, H1p --> Z1p + H1p
    # 1.0, H1m --> Z1m + H1m
    1.0, H2p --> Z2p + H2p
    # 1.0, H2m --> Z2m + H2m

    # 100.0, Z1p + Z1m --> 0
    # 100.0, Z2p + Z2m --> 0
    # 1.0, 2Z1p --> Z1p
    # 1.0, 2Z2p --> Z2p
    # 1.0, 2Z1m --> Z1m
    # 1.0, 2Z2m --> Z2m
end
rn_dual_node_bwd = @reaction_network rn_dual_node_bwd begin
    # dz/dt = -p_i x_i
    # 1.0, P1p + X1m --> Z1p + P1p + X1m
    # 1.0, P1m + X1p --> Z1p + P1m + X1p
    # 1.0, P1p + X1p --> Z1m + P1p + X1p
    # 1.0, P1m + X1m --> Z1m + P1m + X1m
    # 1.0, P2p + X2m --> Z2p + P2p + X2m
    # 1.0, P2m + X2p --> Z2p + P2m + X2p
    # 1.0, P2p + X2p --> Z2m + P2p + X2p
    # 1.0, P2m + X2m --> Z2m + P2m + X2m

    # 1.0, H1m --> Z1p + H1m
    # 1.0, H1p --> Z1m + H1p
    # 1.0, H2m --> Z2p + H2m
    # 1.0, H2p --> Z2m + H2p
    
    # dg_i/dt = a_i x_i
    1.0, A1p + X1p --> G1p + A1p + X1p
    # 1.0, A1m + X1m --> G1p + A1m + X1m
    # 1.0, A1p + X1m --> G1m + A1p + X1m
    1.0, A1m + X1p --> G1m + A1m + X1p
    1.0, A2p + X2p --> G2p + A2p + X2p
    # 1.0, A2m + X2m --> G2p + A2m + X2m
    # 1.0, A2p + X2m --> G2m + A2p + X2m
    1.0, A2m + X2p --> G2m + A2m + X2p
end

rn_dual_mult = @reaction_network rn_dual_mult begin
    1.0, A1p + B1p --> Yp + A1p + B1p
    1.0, A1m + B1m --> Yp + A1m + B1m
    1.0, A1p + B1m --> Ym + A1p + B1m
    1.0, A1m + B1p --> Ym + A1m + B1p
    1.0, A2p + B2p --> Yp + A2p + B2p
    1.0, A2m + B2m --> Yp + A2m + B2m
    1.0, A2p + B2m --> Ym + A2p + B2m
    1.0, A2m + B2p --> Ym + A2m + B2p
end

rn_param_update = @reaction_network rn_param_update begin
    k1, G1p --> P1m
    k1, G1m --> P1p
    k1, G2p --> P2m
    k1, G2m --> P2p
    k2, G1p --> 0
    k2, G2p --> 0
    k2, G1m --> 0
    k2, G2m --> 0
end

rn_final_layer_update = @reaction_network rn_final_layer_update begin
    k1, M1p --> W1m
    k1, M1m --> W1p
    k1, M2p --> W2m
    k1, M2m --> W2p
    k2, M1p --> 0
    k2, M1m --> 0
    k2, M2p --> 0
    k2, M2m --> 0
end

rn_dissipate_reactions = @reaction_network rn_dissipate_reactions begin
    1.0, G1p --> 0
    1.0, G1m --> 0
    1.0, G2p --> 0
    1.0, G2m --> 0
    1.0, M1p --> 0
    1.0, M1m --> 0
    1.0, M2p --> 0
    1.0, M2m --> 0
    1.0, A1p --> 0
    1.0, A1m --> 0
    1.0, A2p --> 0
    1.0, A2m --> 0
    1.0, Ep --> 0
    1.0, Em --> 0
    1.0, Op --> 0
    1.0, Om --> 0
    1.0, Yp --> 0
    1.0, Ym --> 0
end

rn_dual_mult = @reaction_network rn_dual_dot begin
    1.0, Ap + Bp --> Yp + Ap + Bp
    1.0, Am + Bm --> Yp + Am + Bm
    1.0, Ap + Bm --> Ym + Ap + Bm
    1.0, Am + Bp --> Ym + Am + Bp
    1.0, Yp --> 0
    1.0, Ym --> 0
end

rn_dual_dot = @reaction_network rn_dual_dot begin
    1.0, A1p + B1p --> Yp + A1p + B1p
    1.0, A1m + B1m --> Yp + A1m + B1m
    1.0, A2p + B2p --> Yp + A2p + B2p
    1.0, A2m + B2m --> Yp + A2m + B2m
    1.0, Yp --> 0
    1.0, A1p + B1m --> Ym + A1p + B1m
    1.0, A1m + B1p --> Ym + A1m + B1p
    1.0, A2p + B2m --> Ym + A2p + B2m
    1.0, A2m + B2p --> Ym + A2m + B2p
    1.0, Ym --> 0
end

rn_dual_subtract = @reaction_network rn_dual_subtract begin
    1.0, Ap --> Yp
    1.0, Am --> Ym
    1.0, Bp --> Ym
    1.0, Bm --> Yp
end

rn_dual_add = @reaction_network rn_dual_add begin
    1.0, Ap --> Yp
    1.0, Bp --> Yp
    1.0, Am --> Ym
    1.0, Bm --> Ym
end

rn_output_annihilation = @reaction_network rn_output_annihilation begin
    1.0, Op + Om --> 0
end


rn_create_error_species = @reaction_network rn_create_error_species begin
    10.0, Op --> Ep
    10.0, Ym --> Ep
    10.0, Om --> Em
    10.0, Yp --> Em
    100.0, Ep + Em --> 0
end

rn_dual_binary_scalar_mult = @reaction_network rn_dual_binary_scalar_mult begin
    1.0, Ep + S1p --> P1p + Ep + S1p
    1.0, Ep + S1m --> P1m + Ep + S1m
    1.0, Ep + S2p --> P2p + Ep + S2p
    1.0, Ep + S2m --> P2m + Ep + S2m
    1.0, Em + S1p --> P1m + Em + S1p
    1.0, Em + S1m --> P1p + Em + S1m
    1.0, Em + S2p --> P2m + Em + S2p
    1.0, Em + S2m --> P2p + Em + S2m
    1.0, P1p --> 0
    1.0, P1m --> 0
    1.0, P2p --> 0
    1.0, P2m --> 0
end