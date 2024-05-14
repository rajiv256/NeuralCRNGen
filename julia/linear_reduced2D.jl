rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
    # dz_i/dt = p_iz_i
    1.0, P1p + Z1p --> Z1p + P1p + Z1p
    1.0, P1m + Z1m --> Z1p + P1m + Z1m
    1.0, P1p + Z1m --> Z1m + P1p + Z1m
    1.0, P1m + Z1p --> Z1m + P1m + Z1p
    1.0, P2p + Z2p --> Z2p + P2p + Z2p
    1.0, P2m + Z2m --> Z2p + P2m + Z2m
    1.0, P2p + Z2m --> Z2m + P2p + Z2m
    1.0, P2m + Z2p --> Z2m + P2m + Z2p
end
rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin
    # dz/dt = -p_i z_i
    1.0, P1p + Z1m --> Z1p + P1p + Z1m
    1.0, P1m + Z1p --> Z1p + P1m + Z1p
    1.0, P1p + Z1p --> Z1m + P1p + Z1p
    1.0, P1m + Z1m --> Z1m + P1m + Z1m
    1.0, P2p + Z2m --> Z2p + P2p + Z2m
    1.0, P2m + Z2p --> Z2p + P2m + Z2p
    1.0, P2p + Z2p --> Z2m + P2p + Z2p
    1.0, P2m + Z2m --> Z2m + P2m + Z2m

    # da_i/dt = a_i p_i
    1.0, A1p + P1m --> A1p + A1p + P1m
    1.0, A1m + P1p --> A1p + A1m + P1p
    1.0, A1p + P1p --> A1m + A1p + P1p
    1.0, A1m + P1m --> A1m + A1m + P1m
    1.0, A2p + P2m --> A2p + A2p + P2m
    1.0, A2m + P2p --> A2p + A2m + P2p
    1.0, A2p + P2p --> A2m + A2p + P2p
    1.0, A2m + P2m --> A2m + A2m + P2m
    # dg_i/dt = a_i z_i
    1.0, A1p + Z1p --> G1p + A1p + Z1p
    1.0, A1m + Z1m --> G1p + A1m + Z1m
    1.0, A1p + Z1m --> G1m + A1p + Z1m
    1.0, A1m + Z1p --> G1m + A1m + Z1p
    1.0, A2p + Z2p --> G2p + A2p + Z2p
    1.0, A2m + Z2m --> G2p + A2m + Z2m
    1.0, A2p + Z2m --> G2m + A2p + Z2m
    1.0, A2m + Z2p --> G2m + A2m + Z2p
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
    k1, G11p --> P11m
    k1, G11m --> P11p
    k1, G12p --> P12m
    k1, G12m --> P12p
    k1, G21p --> P21m
    k1, G21m --> P21p
    k1, G22p --> P22m
    k1, G22m --> P22p
    k2, G11p --> 0
    k2, G11m --> 0
    k2, G12p --> 0
    k2, G12m --> 0
    k2, G21p --> 0
    k2, G21m --> 0
    k2, G22p --> 0
    k2, G22m --> 0
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
    1.0, G11p --> 0
    1.0, G11m --> 0
    1.0, G12p --> 0
    1.0, G12m --> 0
    1.0, G21p --> 0
    1.0, G21m --> 0
    1.0, G22p --> 0
    1.0, G22m --> 0
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
    1.0, Op --> Ep + Op
    1.0, Ym --> Ep + Ym
    1.0, Om --> Em + Om
    1.0, Yp --> Em + Yp
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