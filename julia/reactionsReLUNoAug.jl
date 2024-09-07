rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
    # dz_i/dt = h
    1.0, H1p --> Z1p + H1p
    1.0, H1m --> Z1m + H1m
    1.0, H2p --> Z2p + H2p
    1.0, H2m --> Z2m + H2m
    # dz_i/dt = p_ij x_j z_i
    1.0, P11p + Z1p + X1p --> Z1p + P11p + Z1p + X1p
    1.0, P11m + Z1m + X1p --> Z1p + P11m + Z1m + X1p
    1.0, P11p + Z1m + X1m --> Z1p + P11p + Z1m + X1m
    1.0, P11m + Z1p + X1m --> Z1p + P11m + Z1p + X1m
    1.0, P11p + Z1p + X1m --> Z1m + P11p + Z1p + X1m
    1.0, P11m + Z1m + X1m --> Z1m + P11m + Z1m + X1m
    1.0, P11p + Z1m + X1p --> Z1m + P11p + Z1m + X1p
    1.0, P11m + Z1p + X1p --> Z1m + P11m + Z1p + X1p


    1.0, P12p + Z1p + X2p --> Z1p + P12p + Z1p + X2p
    1.0, P12m + Z1m + X2p --> Z1p + P12m + Z1m + X2p
    1.0, P12p + Z1m + X2m --> Z1p + P12p + Z1m + X2m
    1.0, P12m + Z1p + X2m --> Z1p + P12m + Z1p + X2m
    1.0, P12p + Z1p + X2m --> Z1m + P12p + Z1p + X2m
    1.0, P12m + Z1m + X2m --> Z1m + P12m + Z1m + X2m
    1.0, P12p + Z1m + X2p --> Z1m + P12p + Z1m + X2p
    1.0, P12m + Z1p + X2p --> Z1m + P12m + Z1p + X2p

    1.0, P21p + Z2p + X1p --> Z2p + P21p + Z2p + X1p
    1.0, P21m + Z2m + X1p --> Z2p + P21m + Z2m + X1p
    1.0, P21p + Z2m + X1m --> Z2p + P21p + Z2m + X1m
    1.0, P21m + Z2p + X1m --> Z2p + P21m + Z2p + X1m
    1.0, P21p + Z2p + X1m --> Z2m + P21p + Z2p + X1m
    1.0, P21m + Z2m + X1m --> Z2m + P21m + Z2m + X1m
    1.0, P21p + Z2m + X1p --> Z2m + P21p + Z2m + X1p
    1.0, P21m + Z2p + X1p --> Z2m + P21m + Z2p + X1p

    1.0, P22p + Z2p + X2p --> Z2p + P22p + Z2p + X2p
    1.0, P22m + Z2m + X2p --> Z2p + P22m + Z2m + X2p
    1.0, P22p + Z2m + X2m --> Z2p + P22p + Z2m + X2m
    1.0, P22m + Z2p + X2m --> Z2p + P22m + Z2p + X2m
    1.0, P22p + Z2p + X2m --> Z2m + P22p + Z2p + X2m
    1.0, P22m + Z2m + X2m --> Z2m + P22m + Z2m + X2m
    1.0, P22p + Z2m + X2p --> Z2m + P22p + Z2m + X2p
    1.0, P22m + Z2p + X2p --> Z2m + P22m + Z2p + X2p

    # dz_i/dt = b_iz_i
    1.0, B1p + Z1p --> Z1p + Z1p + B1p
    1.0, B1m + Z1m --> Z1p + Z1m + B1m
    1.0, B1p + Z1m --> Z1m + Z1m + B1p
    1.0, B1m + Z1p --> Z1m + Z1p + B1m

    1.0, B2p + Z2p --> Z2p + Z2p + B2p
    1.0, B2m + Z2m --> Z2p + Z2m + B2m
    1.0, B2p + Z2m --> Z2m + Z2m + B2p
    1.0, B2m + Z2p --> Z2m + Z2p + B2m

    # dz_i/dt = -z_i^2 
    # Not sure if this is hacky
    1.0, Z1p + Z1p --> Z1p
    1.0, Z1m + Z1m --> Z1m
    100.0, Z1p + Z1m --> 0

    1.0, Z2p + Z2p --> Z2p
    1.0, Z2m + Z2m --> Z2m
    100.0, Z2p + Z2m --> 0

end


rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin
    # dz/dt = -h
    1.0, H1m --> Z1p + H1m
    1.0, H1p --> Z1m + H1p
    1.0, H1m --> Z2p + H1m
    1.0, H1p --> Z2m + H1p
    # dz/dt = -p_ij x_j z_i
    1.0, P11p + Z1p + X1m --> Z1p + P11p + Z1p + X1m
    1.0, P11m + Z1m + X1m --> Z1p + P11m + Z1m + X1m
    1.0, P11p + Z1m + X1p --> Z1p + P11p + Z1m + X1p
    1.0, P11m + Z1p + X1p --> Z1p + P11m + Z1p + X1p
    1.0, P11p + Z1p + X1p --> Z1m + P11p + Z1p + X1p
    1.0, P11m + Z1m + X1p --> Z1m + P11m + Z1m + X1p
    1.0, P11p + Z1m + X1m --> Z1m + P11p + Z1m + X1m
    1.0, P11m + Z1p + X1m --> Z1m + P11m + Z1p + X1m
    1.0, P12p + Z1p + X2m --> Z1p + P12p + Z1p + X2m
    1.0, P12m + Z1m + X2m --> Z1p + P12m + Z1m + X2m
    1.0, P12p + Z1m + X2p --> Z1p + P12p + Z1m + X2p
    1.0, P12m + Z1p + X2p --> Z1p + P12m + Z1p + X2p
    1.0, P12p + Z1p + X2p --> Z1m + P12p + Z1p + X2p
    1.0, P12m + Z1m + X2p --> Z1m + P12m + Z1m + X2p
    1.0, P12p + Z1m + X2m --> Z1m + P12p + Z1m + X2m
    1.0, P12m + Z1p + X2m --> Z1m + P12m + Z1p + X2m
    1.0, P21p + Z2p + X1m --> Z2p + P21p + Z2p + X1m
    1.0, P21m + Z2m + X1m --> Z2p + P21m + Z2m + X1m
    1.0, P21p + Z2m + X1p --> Z2p + P21p + Z2m + X1p
    1.0, P21m + Z2p + X1p --> Z2p + P21m + Z2p + X1p
    1.0, P21p + Z2p + X1p --> Z2m + P21p + Z2p + X1p
    1.0, P21m + Z2m + X1p --> Z2m + P21m + Z2m + X1p
    1.0, P21p + Z2m + X1m --> Z2m + P21p + Z2m + X1m
    1.0, P21m + Z2p + X1m --> Z2m + P21m + Z2p + X1m
    1.0, P22p + Z2p + X2m --> Z2p + P22p + Z2p + X2m
    1.0, P22m + Z2m + X2m --> Z2p + P22m + Z2m + X2m
    1.0, P22p + Z2m + X2p --> Z2p + P22p + Z2m + X2p
    1.0, P22m + Z2p + X2p --> Z2p + P22m + Z2p + X2p
    1.0, P22p + Z2p + X2p --> Z2m + P22p + Z2p + X2p
    1.0, P22m + Z2m + X2p --> Z2m + P22m + Z2m + X2p
    1.0, P22p + Z2m + X2m --> Z2m + P22p + Z2m + X2m
    1.0, P22m + Z2p + X2m --> Z2m + P22m + Z2p + X2m
    
    # dz/dt = -b_i z_i
    1.0, B1p + Z1m --> Z1p + B1p + Z1m
    1.0, B1m + Z1p --> Z1p + B1m + Z1p
    1.0, B1p + Z1p --> Z1m + B1p + Z1p
    1.0, B1m + Z1m --> Z1m + B1m + Z1m
    1.0, B2p + Z2m --> Z2p + B2p + Z2m
    1.0, B2m + Z2p --> Z2p + B2m + Z2p
    1.0, B2p + Z2p --> Z2m + B2p + Z2p
    1.0, B2m + Z2m --> Z2m + B2m + Z2m

    # dz/dt = z^2
    10.0, Z1p + Z1m --> 0
    1.0, Z1p + Z1p --> Z1p + Z1p + Z1p
    1.0, Z1m + Z1m --> Z1m + Z1m + Z1m

    10.0, Z2p + Z2m --> 0
    1.0, Z2p + Z2p --> Z2p + Z2p + Z2p
    1.0, Z2m + Z2m --> Z2m + Z2m + Z2m

    # da_i/dt = a_i p_ij x_j
    1.0, A1p + P11p + X1p --> A1p + A1p + P11p + X1p
    1.0, A1m + P11m + X1p --> A1p + A1m + P11m + X1p
    1.0, A1p + P11m + X1m --> A1p + A1p + P11m + X1m
    1.0, A1m + P11p + X1m --> A1p + A1m + P11p + X1m
    1.0, A1p + P11p + X1m --> A1m + A1p + P11p + X1m
    1.0, A1m + P11m + X1m --> A1m + A1m + P11m + X1m
    1.0, A1p + P11m + X1p --> A1m + A1p + P11m + X1p
    1.0, A1m + P11p + X1p --> A1m + A1m + P11p + X1p
    1.0, A1p + P12p + X2p --> A1p + A1p + P12p + X2p
    1.0, A1m + P12m + X2p --> A1p + A1m + P12m + X2p
    1.0, A1p + P12m + X2m --> A1p + A1p + P12m + X2m
    1.0, A1m + P12p + X2m --> A1p + A1m + P12p + X2m
    1.0, A1p + P12p + X2m --> A1m + A1p + P12p + X2m
    1.0, A1m + P12m + X2m --> A1m + A1m + P12m + X2m
    1.0, A1p + P12m + X2p --> A1m + A1p + P12m + X2p
    1.0, A1m + P12p + X2p --> A1m + A1m + P12p + X2p
    
    1.0, A2p + P21p + X1p --> A2p + A2p + P21p + X1p
    1.0, A2m + P21m + X1p --> A2p + A2m + P21m + X1p
    1.0, A2p + P21m + X1m --> A2p + A2p + P21m + X1m
    1.0, A2m + P21p + X1m --> A2p + A2m + P21p + X1m
    1.0, A2p + P21p + X1m --> A2m + A2p + P21p + X1m
    1.0, A2m + P21m + X1m --> A2m + A2m + P21m + X1m
    1.0, A2p + P21m + X1p --> A2m + A2p + P21m + X1p
    1.0, A2m + P21p + X1p --> A2m + A2m + P21p + X1p
    1.0, A2p + P22p + X2p --> A2p + A2p + P22p + X2p
    1.0, A2m + P22m + X2p --> A2p + A2m + P22m + X2p
    1.0, A2p + P22m + X2m --> A2p + A2p + P22m + X2m
    1.0, A2m + P22p + X2m --> A2p + A2m + P22p + X2m
    1.0, A2p + P22p + X2m --> A2m + A2p + P22p + X2m
    1.0, A2m + P22m + X2m --> A2m + A2m + P22m + X2m
    1.0, A2p + P22m + X2p --> A2m + A2p + P22m + X2p
    1.0, A2m + P22p + X2p --> A2m + A2m + P22p + X2p


    # da_i/dt = -2 a_i z_i
    2.0, A1p + Z1m --> A1p + A1p + Z1m
    2.0, A1m + Z1p --> A1p + A1m + Z1p
    2.0, A1p + Z1p --> A1m + A1p + Z1p
    2.0, A1m + Z1m --> A1m + A1m + Z1m
    2.0, A2p + Z2m --> A2p + A2p + Z2m
    2.0, A2m + Z2p --> A2p + A2m + Z2p
    2.0, A2p + Z2p --> A2m + A2p + Z2p
    2.0, A2m + Z2m --> A2m + A2m + Z2m

    # dg_ij/dt = a_i z_i x_j
    1.0, A1p + Z1p + X1p --> G11p + A1p + Z1p + X1p
    1.0, A1m + Z1m + X1p --> G11p + A1m + Z1m + X1p
    1.0, A1p + Z1m + X1m --> G11p + A1p + Z1m + X1m
    1.0, A1m + Z1p + X1m --> G11p + A1m + Z1p + X1m
    1.0, A1p + Z1p + X1m --> G11m + A1p + Z1p + X1m
    1.0, A1m + Z1m + X1m --> G11m + A1m + Z1m + X1m
    1.0, A1p + Z1m + X1p --> G11m + A1p + Z1m + X1p
    1.0, A1m + Z1p + X1p --> G11m + A1m + Z1p + X1p
    1.0, A1p + Z1p + X2p --> G12p + A1p + Z1p + X2p
    1.0, A1m + Z1m + X2p --> G12p + A1m + Z1m + X2p
    1.0, A1p + Z1m + X2m --> G12p + A1p + Z1m + X2m
    1.0, A1m + Z1p + X2m --> G12p + A1m + Z1p + X2m
    1.0, A1p + Z1p + X2m --> G12m + A1p + Z1p + X2m
    1.0, A1m + Z1m + X2m --> G12m + A1m + Z1m + X2m
    1.0, A1p + Z1m + X2p --> G12m + A1p + Z1m + X2p
    1.0, A1m + Z1p + X2p --> G12m + A1m + Z1p + X2p

    1.0, A2p + Z2p + X1p --> G21p + A2p + Z2p + X1p
    1.0, A2m + Z2m + X1p --> G21p + A2m + Z2m + X1p
    1.0, A2p + Z2m + X1m --> G21p + A2p + Z2m + X1m
    1.0, A2m + Z2p + X1m --> G21p + A2m + Z2p + X1m
    1.0, A2p + Z2p + X1m --> G21m + A2p + Z2p + X1m
    1.0, A2m + Z2m + X1m --> G21m + A2m + Z2m + X1m
    1.0, A2p + Z2m + X1p --> G21m + A2p + Z2m + X1p
    1.0, A2m + Z2p + X1p --> G21m + A2m + Z2p + X1p
    1.0, A2p + Z2p + X2p --> G22p + A2p + Z2p + X2p
    1.0, A2m + Z2m + X2p --> G22p + A2m + Z2m + X2p
    1.0, A2p + Z2m + X2m --> G22p + A2p + Z2m + X2m
    1.0, A2m + Z2p + X2m --> G22p + A2m + Z2p + X2m
    1.0, A2p + Z2p + X2m --> G22m + A2p + Z2p + X2m
    1.0, A2m + Z2m + X2m --> G22m + A2m + Z2m + X2m
    1.0, A2p + Z2m + X2p --> G22m + A2p + Z2m + X2p
    1.0, A2m + Z2p + X2p --> G22m + A2m + Z2p + X2p


    # dbg_i/dt = a_i z_i
    1.0, A1p + Z1p --> V1p + A1p + Z1p
    1.0, A1m + Z1m --> V1p + A1m + Z1m
    1.0, A1p + Z1m --> V1m + A1p + Z1m
    1.0, A1m + Z1p --> V1m + A1m + Z1p
    1.0, A2p + Z2p --> V2p + A2p + Z2p
    1.0, A2m + Z2m --> V2p + A2m + Z2m
    1.0, A2p + Z2m --> V2m + A2p + Z2m
    1.0, A2m + Z2p --> V2m + A2m + Z2p

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
    
    
    k1, V1p --> B1m
    k1, V1m --> B1p
    k1, V2p --> B2m
    k1, V2m --> B2p
   
    k2, V1p --> 0
    k2, V1m --> 0
    k2, V2p --> 0
    k2, V2m --> 0
   
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
    1.0, Em + S1p --> P1m + Em + S1p
    1.0, Em + S1m --> P1p + Em + S1m
    1.0, Em + S2p --> P2m + Em + S2p
    1.0, Em + S2m --> P2p + Em + S2m
   
    1.0, P1p --> 0
    1.0, P1m --> 0
    1.0, P2p --> 0
    1.0, P2m --> 0
end

