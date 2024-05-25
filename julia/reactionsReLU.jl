rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
    # dz_i/dt = h
    1.0, H1p --> Z1p + H1p
    1.0, H1m --> Z1m + H1m
    1.0, H1p --> Z2p + H1p
    1.0, H1m --> Z2m + H1m
    1.0, H1p --> Z3p + H1p
    1.0, H1m --> Z3m + H1m
    # dz_i/dt = p_iz_i
    1.0, P1p + Z1p --> Z1p + P1p + Z1p
    1.0, P1m + Z1m --> Z1p + P1m + Z1m
    1.0, P1p + Z1m --> Z1m + P1p + Z1m
    1.0, P1m + Z1p --> Z1m + P1m + Z1p
    1.0, P2p + Z2p --> Z2p + P2p + Z2p
    1.0, P2m + Z2m --> Z2p + P2m + Z2m
    1.0, P2p + Z2m --> Z2m + P2p + Z2m
    1.0, P2m + Z2p --> Z2m + P2m + Z2p
    1.0, P3p + Z3p --> Z3p + P3p + Z3p
    1.0, P3m + Z3m --> Z3p + P3m + Z3m
    1.0, P3p + Z3m --> Z3m + P3p + Z3m
    1.0, P3m + Z3p --> Z3m + P3m + Z3p

    # dz_i/dt = -z_i^2 
    # Not sure if this is hacky
    1.0, Z1p + Z1p --> Z1p 
    1.0, Z1m + Z1m --> Z1m 
    100.0, Z1p + Z1m --> 0

    1.0, Z2p + Z2p --> Z2p
    1.0, Z2m + Z2m --> Z2m
    100.0, Z2p + Z2m --> 0

    1.0, Z3p + Z3p --> Z3p
    1.0, Z3m + Z3m --> Z3m
    100.0, Z3p + Z3m --> 0
end


rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin
    # dz/dt = -h
    1.0, H1m --> Z1p + H1m
    1.0, H1p --> Z1m + H1p
    1.0, H1m --> Z2p + H1m
    1.0, H1p --> Z2m + H1p
    1.0, H1m --> Z3p + H1m
    1.0, H1p --> Z3m + H1p
    # dz/dt = -p_iz_i
    1.0, P1p + Z1m --> Z1p + P1p + Z1m
    1.0, P1m + Z1p --> Z1p + P1m + Z1p
    1.0, P1p + Z1p --> Z1m + P1p + Z1p
    1.0, P1m + Z1m --> Z1m + P1m + Z1m
    1.0, P2p + Z2m --> Z2p + P2p + Z2m
    1.0, P2m + Z2p --> Z2p + P2m + Z2p
    1.0, P2p + Z2p --> Z2m + P2p + Z2p
    1.0, P2m + Z2m --> Z2m + P2m + Z2m
    1.0, P3p + Z3m --> Z3p + P3p + Z3m
    1.0, P3m + Z3p --> Z3p + P3m + Z3p
    1.0, P3p + Z3p --> Z3m + P3p + Z3p
    1.0, P3m + Z3m --> Z3m + P3m + Z3m

    # dz/dt = z^2
    10.0, Z1p + Z1m --> 0
    1.0, Z1p + Z1p --> Z1p + Z1p + Z1p
    1.0, Z1m + Z1m --> Z1m + Z1m + Z1m

    10.0, Z2p + Z2m --> 0
    1.0, Z2p + Z2p --> Z2p + Z2p + Z2p
    1.0, Z2m + Z2m --> Z2m + Z2m + Z2m

    10.0, Z3p + Z3m --> 0
    1.0, Z3p + Z3p --> Z3p + Z3p + Z3p
    1.0, Z3m + Z3m --> Z3m + Z3m + Z3m

    # da/dt = a_ip_i
    1.0, A1p + P1p --> A1p + A1p + P1p
    1.0, A1m + P1m --> A1p + A1m + P1m
    1.0, A1p + P1m --> A1m + A1p + P1m
    1.0, A1m + P1p --> A1m + A1m + P1p
    1.0, A2p + P2p --> A2p + A2p + P2p
    1.0, A2m + P2m --> A2p + A2m + P2m
    1.0, A2p + P2m --> A2m + A2p + P2m
    1.0, A2m + P2p --> A2m + A2m + P2p
    1.0, A3p + P3p --> A3p + A3p + P3p
    1.0, A3m + P3m --> A3p + A3m + P3m
    1.0, A3p + P3m --> A3m + A3p + P3m
    1.0, A3m + P3p --> A3m + A3m + P3p

    # dg_i/dt = a_i z_i
    1.0, A1p + Z1p --> G1p + A1p + Z1p
    1.0, A1m + Z1m --> G1p + A1m + Z1m
    1.0, A1p + Z1m --> G1m + A1p + Z1m
    1.0, A1m + Z1p --> G1m + A1m + Z1p
    1.0, A2p + Z2p --> G2p + A2p + Z2p
    1.0, A2m + Z2m --> G2p + A2m + Z2m
    1.0, A2p + Z2m --> G2m + A2p + Z2m
    1.0, A2m + Z2p --> G2m + A2m + Z2p
    1.0, A3p + Z3p --> G3p + A3p + Z3p
    1.0, A3m + Z3m --> G3p + A3m + Z3m
    1.0, A3p + Z3m --> G3m + A3p + Z3m
    1.0, A3m + Z3p --> G3m + A3m + Z3p

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
    k1, G3p --> P3m
    k1, G3m --> P3p
    k2, G1p --> 0
    k2, G1m --> 0
    k2, G2p --> 0
    k2, G3p --> 0
    k2, G3m --> 0
end


rn_final_layer_update = @reaction_network rn_final_layer_update begin
    k1, M1p --> W1m
    k1, M1m --> W1p
    k1, M2p --> W2m
    k1, M2m --> W2p
    k1, M3p --> W3m
    k1, M3m --> W3p
    k2, M1p --> 0
    k2, M1m --> 0
    k2, M2p --> 0
    k2, M2m --> 0
    k2, M3p --> 0
    k2, M3m --> 0
end


rn_dual_mult = @reaction_network rn_dual_mult begin
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
    1.0, A3p + B3p --> Yp + A3p + B3p
    1.0, A3m + B3m --> Yp + A3m + B3m
    1.0, Yp --> 0
    1.0, A1p + B1m --> Ym + A1p + B1m
    1.0, A1m + B1p --> Ym + A1m + B1p
    1.0, A2p + B2m --> Ym + A2p + B2m
    1.0, A2m + B2p --> Ym + A2m + B2p
    1.0, A3p + B3m --> Ym + A3p + B3m
    1.0, A3m + B3p --> Ym + A3m + B3p
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
    # 1.0, Op + Ym --> Ep + Op + Ym
    # 1.0, Om + Yp --> Em + Om + Yp
    10.0, Op --> Ep
    10.0, Yp --> Em
    10.0, Om --> Em
    10.0, Ym --> Ep 
    100.0, Ep + Em --> 0
end


rn_dual_binary_scalar_mult = @reaction_network rn_dual_binary_scalar_mult begin
    1.0, Ep + S1p --> P1p + Ep + S1p 
    1.0, Ep + S1m --> P1m + Ep + S1m 
    1.0, Ep + S2p --> P2p + Ep + S2p
    1.0, Ep + S2m --> P2m + Ep + S2m
    1.0, Ep + S3p --> P3p + Ep + S3p
    1.0, Ep + S3m --> P3m + Ep + S3m
    1.0, Em + S1p --> P1m + Em + S1p
    1.0, Em + S1m --> P1p + Em + S1m
    1.0, Em + S2p --> P2m + Em + S2p
    1.0, Em + S2m --> P2p + Em + S2m
    1.0, Em + S3p --> P3m + Em + S3p
    1.0, Em + S3m --> P3p + Em + S3m
    1.0, P1p --> 0
    1.0, P1m --> 0
    1.0, P2p --> 0
    1.0, P2m --> 0
    1.0, P3p --> 0
    1.0, P3m --> 0
end

