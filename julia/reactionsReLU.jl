rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
    # dz_i/dt = h
    1.0, H1p --> Z1p + H1p
    1.0, H1m --> Z1m + H1m
    1.0, H1p --> Z2p + H1p
    1.0, H1m --> Z2m + H1m
    1.0, H1p --> Z3p + H1p
    1.0, H1m --> Z3m + H1m
    # dz_i/dt = p_i x_i
    1.0, P11p + X1p --> Z1p + P11p + X1p
    1.0, P11m + X1m --> Z1p + P11m + X1m
    1.0, P11p + X1m --> Z1m + P11p + X1m
    1.0, P11m + X1p --> Z1m + P11m + X1p
    1.0, P12p + X2p --> Z1p + P12p + X2p
    1.0, P12m + X2m --> Z1p + P12m + X2m
    1.0, P12p + X2m --> Z1m + P12p + X2m
    1.0, P12m + X2p --> Z1m + P12m + X2p
    1.0, P13p + X3p --> Z1p + P13p + X3p
    1.0, P13m + X3m --> Z1p + P13m + X3m
    1.0, P13p + X3m --> Z1m + P13p + X3m
    1.0, P13m + X3p --> Z1m + P13m + X3p
    1.0, P21p + X1p --> Z2p + P21p + X1p
    1.0, P21m + X1m --> Z2p + P21m + X1m
    1.0, P21p + X1m --> Z2m + P21p + X1m
    1.0, P21m + X1p --> Z2m + P21m + X1p
    1.0, P22p + X2p --> Z2p + P22p + X2p
    1.0, P22m + X2m --> Z2p + P22m + X2m
    1.0, P22p + X2m --> Z2m + P22p + X2m
    1.0, P22m + X2p --> Z2m + P22m + X2p
    1.0, P23p + X3p --> Z2p + P23p + X3p
    1.0, P23m + X3m --> Z2p + P23m + X3m
    1.0, P23p + X3m --> Z2m + P23p + X3m
    1.0, P23m + X3p --> Z2m + P23m + X3p
    1.0, P31p + X1p --> Z3p + P31p + X1p
    1.0, P31m + X1m --> Z3p + P31m + X1m
    1.0, P31p + X1m --> Z3m + P31p + X1m
    1.0, P31m + X1p --> Z3m + P31m + X1p
    1.0, P32p + X2p --> Z3p + P32p + X2p
    1.0, P32m + X2m --> Z3p + P32m + X2m
    1.0, P32p + X2m --> Z3m + P32p + X2m
    1.0, P32m + X2p --> Z3m + P32m + X2p
    1.0, P33p + X3p --> Z3p + P33p + X3p
    1.0, P33m + X3m --> Z3p + P33m + X3m
    1.0, P33p + X3m --> Z3m + P33p + X3m
    1.0, P33m + X3p --> Z3m + P33m + X3p

    # dz_i/dt = -z_i^3
    # Not sure if this is hacky
    1.0, 3Z1p --> 2Z1p 
    1.0, 3Z1m --> 2Z1m 
    100.0, Z1p + Z1m --> 0

    1.0, 3Z2p --> 2Z2p
    1.0, 3Z2m --> 2Z2m
    100.0, Z2p + Z2m --> 0

    1.0, 3Z3p --> 2Z3p
    1.0, 3Z3m --> 2Z3m
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
    # dz/dt = -p_i x_i
    1.0, P11p + X1m --> Z1p + P11p + X1m
    1.0, P11m + X1p --> Z1p + P11m + X1p
    1.0, P11p + X1p --> Z1m + P11p + X1p
    1.0, P11m + X1m --> Z1m + P11m + X1m
    1.0, P12p + X2m --> Z1p + P12p + X2m
    1.0, P12m + X2p --> Z1p + P12m + X2p
    1.0, P12p + X2p --> Z1m + P12p + X2p
    1.0, P12m + X2m --> Z1m + P12m + X2m
    1.0, P13p + X3m --> Z1p + P13p + X3m
    1.0, P13m + X3p --> Z1p + P13m + X3p
    1.0, P13p + X3p --> Z1m + P13p + X3p
    1.0, P13m + X3m --> Z1m + P13m + X3m
    1.0, P21p + X1m --> Z2p + P21p + X1m
    1.0, P21m + X1p --> Z2p + P21m + X1p
    1.0, P21p + X1p --> Z2m + P21p + X1p
    1.0, P21m + X1m --> Z2m + P21m + X1m
    1.0, P22p + X2m --> Z2p + P22p + X2m
    1.0, P22m + X2p --> Z2p + P22m + X2p
    1.0, P22p + X2p --> Z2m + P22p + X2p
    1.0, P22m + X2m --> Z2m + P22m + X2m
    1.0, P23p + X3m --> Z2p + P23p + X3m
    1.0, P23m + X3p --> Z2p + P23m + X3p
    1.0, P23p + X3p --> Z2m + P23p + X3p
    1.0, P23m + X3m --> Z2m + P23m + X3m
    1.0, P31p + X1m --> Z3p + P31p + X1m
    1.0, P31m + X1p --> Z3p + P31m + X1p
    1.0, P31p + X1p --> Z3m + P31p + X1p
    1.0, P31m + X1m --> Z3m + P31m + X1m
    1.0, P32p + X2m --> Z3p + P32p + X2m
    1.0, P32m + X2p --> Z3p + P32m + X2p
    1.0, P32p + X2p --> Z3m + P32p + X2p
    1.0, P32m + X2m --> Z3m + P32m + X2m
    1.0, P33p + X3m --> Z3p + P33p + X3m
    1.0, P33m + X3p --> Z3p + P33m + X3p
    1.0, P33p + X3p --> Z3m + P33p + X3p
    1.0, P33m + X3m --> Z3m + P33m + X3m

    # dz_i/dt = -z_i^3
    # Not sure if this is hacky
    1.0, 3Z1p --> 4Z1p
    1.0, 3Z1m --> 4Z1m
    100.0, Z1p + Z1m --> 0 

    1.0, 3Z2p --> 4Z2p
    1.0, 3Z2m --> 4Z2m
    100.0, Z2p + Z2m --> 0

    1.0, 3Z3p --> 4Z3p
    1.0, 3Z3m --> 4Z3m
    100.0, Z3p + Z3m --> 0

    # da_i/dt = - a_i z_i z_i
    1.0, A1p + Z1p + Z1m --> A1p + A1p + Z1p + Z1m
    1.0, A1m + Z1m + Z1m --> A1p + A1m + Z1m + Z1m
    1.0, A1p + Z1m + Z1p --> A1p + A1p + Z1m + Z1p
    1.0, A1m + Z1p + Z1p --> A1p + A1m + Z1p + Z1p
    1.0, A1p + Z1p + Z1p --> A1m + A1p + Z1p + Z1p
    1.0, A1m + Z1m + Z1p --> A1m + A1m + Z1m + Z1p
    1.0, A1p + Z1m + Z1m --> A1m + A1p + Z1m + Z1m
    1.0, A1m + Z1p + Z1m --> A1m + A1m + Z1p + Z1m
    1.0, A2p + Z2p + Z2m --> A2p + A2p + Z2p + Z2m
    1.0, A2m + Z2m + Z2m --> A2p + A2m + Z2m + Z2m
    1.0, A2p + Z2m + Z2p --> A2p + A2p + Z2m + Z2p
    1.0, A2m + Z2p + Z2p --> A2p + A2m + Z2p + Z2p
    1.0, A2p + Z2p + Z2p --> A2m + A2p + Z2p + Z2p
    1.0, A2m + Z2m + Z2p --> A2m + A2m + Z2m + Z2p
    1.0, A2p + Z2m + Z2m --> A2m + A2p + Z2m + Z2m
    1.0, A2m + Z2p + Z2m --> A2m + A2m + Z2p + Z2m
    1.0, A3p + Z3p + Z3m --> A3p + A3p + Z3p + Z3m
    1.0, A3m + Z3m + Z3m --> A3p + A3m + Z3m + Z3m
    1.0, A3p + Z3m + Z3p --> A3p + A3p + Z3m + Z3p
    1.0, A3m + Z3p + Z3p --> A3p + A3m + Z3p + Z3p
    1.0, A3p + Z3p + Z3p --> A3m + A3p + Z3p + Z3p
    1.0, A3m + Z3m + Z3p --> A3m + A3m + Z3m + Z3p
    1.0, A3p + Z3m + Z3m --> A3m + A3p + Z3m + Z3m
    1.0, A3m + Z3p + Z3m --> A3m + A3m + Z3p + Z3m
    # dg_ij/dt = a x
    1.0, A1p + X1p --> G11p + A1p + X1p
    1.0, A1m + X1m --> G11p + A1m + X1m
    1.0, A1p + X1m --> G11m + A1p + X1m
    1.0, A1m + X1p --> G11m + A1m + X1p
    1.0, A1p + X2p --> G12p + A1p + X2p
    1.0, A1m + X2m --> G12p + A1m + X2m
    1.0, A1p + X2m --> G12m + A1p + X2m
    1.0, A1m + X2p --> G12m + A1m + X2p
    1.0, A1p + X3p --> G13p + A1p + X3p
    1.0, A1m + X3m --> G13p + A1m + X3m
    1.0, A1p + X3m --> G13m + A1p + X3m
    1.0, A1m + X3p --> G13m + A1m + X3p
    1.0, A2p + X1p --> G21p + A2p + X1p
    1.0, A2m + X1m --> G21p + A2m + X1m
    1.0, A2p + X1m --> G21m + A2p + X1m
    1.0, A2m + X1p --> G21m + A2m + X1p
    1.0, A2p + X2p --> G22p + A2p + X2p
    1.0, A2m + X2m --> G22p + A2m + X2m
    1.0, A2p + X2m --> G22m + A2p + X2m
    1.0, A2m + X2p --> G22m + A2m + X2p
    1.0, A2p + X3p --> G23p + A2p + X3p
    1.0, A2m + X3m --> G23p + A2m + X3m
    1.0, A2p + X3m --> G23m + A2p + X3m
    1.0, A2m + X3p --> G23m + A2m + X3p
    1.0, A3p + X1p --> G31p + A3p + X1p
    1.0, A3m + X1m --> G31p + A3m + X1m
    1.0, A3p + X1m --> G31m + A3p + X1m
    1.0, A3m + X1p --> G31m + A3m + X1p
    1.0, A3p + X2p --> G32p + A3p + X2p
    1.0, A3m + X2m --> G32p + A3m + X2m
    1.0, A3p + X2m --> G32m + A3p + X2m
    1.0, A3m + X2p --> G32m + A3m + X2p
    1.0, A3p + X3p --> G33p + A3p + X3p
    1.0, A3m + X3m --> G33p + A3m + X3m
    1.0, A3p + X3m --> G33m + A3p + X3m
    1.0, A3m + X3p --> G33m + A3m + X3p
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
    k1, G13p --> P13m
    k1, G13m --> P13p
    k1, G21p --> P21m
    k1, G21m --> P21p
    k1, G22p --> P22m
    k1, G22m --> P22p
    k1, G23p --> P23m
    k1, G23m --> P23p
    k1, G31p --> P31m
    k1, G31m --> P31p
    k1, G32p --> P32m
    k1, G32m --> P32p
    k1, G33p --> P33m
    k1, G33m --> P33p
    k2, G11p --> 0
    k2, G11m --> 0
    k2, G12p --> 0
    k2, G12m --> 0
    k2, G13p --> 0
    k2, G13m --> 0
    k2, G21p --> 0
    k2, G21m --> 0
    k2, G22p --> 0
    k2, G22m --> 0
    k2, G23p --> 0
    k2, G23m --> 0
    k2, G31p --> 0
    k2, G31m --> 0
    k2, G32p --> 0
    k2, G32m --> 0
    k2, G33p --> 0
    k2, G33m --> 0
    k1, V1p --> B1m
    k1, V1m --> B1p
    k1, V2p --> B2m
    k1, V2m --> B2p
    k1, V3p --> B3m
    k1, V3m --> B3p
    k2, V1p --> 0
    k2, V1m --> 0
    k2, V2p --> 0
    k2, V2m --> 0
    k2, V3p --> 0
    k2, V3m --> 0
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

rn_dissipate_reactions = @reaction_network rn_dissipate_reactions begin
    1.0, G11p --> 0
    1.0, G11m --> 0
    1.0, G12p --> 0
    1.0, G12m --> 0
    1.0, G13p --> 0
    1.0, G13m --> 0
    1.0, G21p --> 0
    1.0, G21m --> 0
    1.0, G22p --> 0
    1.0, G22m --> 0
    1.0, G23p --> 0
    1.0, G23m --> 0
    1.0, G31p --> 0
    1.0, G31m --> 0
    1.0, G32p --> 0
    1.0, G32m --> 0
    1.0, G33p --> 0
    1.0, G33m --> 0
    1.0, M1p --> 0
    1.0, M1m --> 0
    1.0, M2p --> 0
    1.0, M2m --> 0
    1.0, M3p --> 0
    1.0, M3m --> 0
    1.0, A1p --> 0
    1.0, A1m --> 0
    1.0, A2p --> 0
    1.0, A2m --> 0
    1.0, A3p --> 0
    1.0, A3m --> 0
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

