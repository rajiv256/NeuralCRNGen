using DifferentialEquations;
using Random;
using Plots;
using Formatting;
using LinearAlgebra;
using Noise;
using ReactionNetworkImporters;
using Dictionaries;
using LaTeXStrings;
using Statistics;
using ColorSchemes;
using Catalyst;

rn_dual_node_fwd = @reaction_network rn_dual_node_fwd begin
    1.0, P11p + Z1p --> Z1p + P11p + Z1p
    1.0, P11m + Z1m --> Z1p + P11m + Z1m
    1.0, P11p + Z1m --> Z1m + P11p + Z1m
    1.0, P11m + Z1p --> Z1m + P11m + Z1p
    1.0, P12p + Z2p --> Z1p + P12p + Z2p
    1.0, P12m + Z2m --> Z1p + P12m + Z2m
    1.0, P12p + Z2m --> Z1m + P12p + Z2m
    1.0, P12m + Z2p --> Z1m + P12m + Z2p
    1.0, P13p + Z3p --> Z1p + P13p + Z3p
    1.0, P13m + Z3m --> Z1p + P13m + Z3m
    1.0, P13p + Z3m --> Z1m + P13p + Z3m
    1.0, P13m + Z3p --> Z1m + P13m + Z3p
    1.0, P21p + Z1p --> Z2p + P21p + Z1p
    1.0, P21m + Z1m --> Z2p + P21m + Z1m
    1.0, P21p + Z1m --> Z2m + P21p + Z1m
    1.0, P21m + Z1p --> Z2m + P21m + Z1p
    1.0, P22p + Z2p --> Z2p + P22p + Z2p
    1.0, P22m + Z2m --> Z2p + P22m + Z2m
    1.0, P22p + Z2m --> Z2m + P22p + Z2m
    1.0, P22m + Z2p --> Z2m + P22m + Z2p
    1.0, P23p + Z3p --> Z2p + P23p + Z3p
    1.0, P23m + Z3m --> Z2p + P23m + Z3m
    1.0, P23p + Z3m --> Z2m + P23p + Z3m
    1.0, P23m + Z3p --> Z2m + P23m + Z3p
    1.0, P31p + Z1p --> Z3p + P31p + Z1p
    1.0, P31m + Z1m --> Z3p + P31m + Z1m
    1.0, P31p + Z1m --> Z3m + P31p + Z1m
    1.0, P31m + Z1p --> Z3m + P31m + Z1p
    1.0, P32p + Z2p --> Z3p + P32p + Z2p
    1.0, P32m + Z2m --> Z3p + P32m + Z2m
    1.0, P32p + Z2m --> Z3m + P32p + Z2m
    1.0, P32m + Z2p --> Z3m + P32m + Z2p
    1.0, P33p + Z3p --> Z3p + P33p + Z3p
    1.0, P33m + Z3m --> Z3p + P33m + Z3m
    1.0, P33p + Z3m --> Z3m + P33p + Z3m
    1.0, P33m + Z3p --> Z3m + P33m + Z3p
end


rn_dual_backprop = @reaction_network rn_dual_backprop begin
    1.0, P11p + Z1m --> Z1p + P11p + Z1m
    1.0, P11m + Z1p --> Z1p + P11m + Z1p
    1.0, P11p + Z1p --> Z1m + P11p + Z1p
    1.0, P11m + Z1m --> Z1m + P11m + Z1m
    1.0, P12p + Z2m --> Z1p + P12p + Z2m
    1.0, P12m + Z2p --> Z1p + P12m + Z2p
    1.0, P12p + Z2p --> Z1m + P12p + Z2p
    1.0, P12m + Z2m --> Z1m + P12m + Z2m
    1.0, P13p + Z3m --> Z1p + P13p + Z3m
    1.0, P13m + Z3p --> Z1p + P13m + Z3p
    1.0, P13p + Z3p --> Z1m + P13p + Z3p
    1.0, P13m + Z3m --> Z1m + P13m + Z3m
    1.0, P21p + Z1m --> Z2p + P21p + Z1m
    1.0, P21m + Z1p --> Z2p + P21m + Z1p
    1.0, P21p + Z1p --> Z2m + P21p + Z1p
    1.0, P21m + Z1m --> Z2m + P21m + Z1m
    1.0, P22p + Z2m --> Z2p + P22p + Z2m
    1.0, P22m + Z2p --> Z2p + P22m + Z2p
    1.0, P22p + Z2p --> Z2m + P22p + Z2p
    1.0, P22m + Z2m --> Z2m + P22m + Z2m
    1.0, P23p + Z3m --> Z2p + P23p + Z3m
    1.0, P23m + Z3p --> Z2p + P23m + Z3p
    1.0, P23p + Z3p --> Z2m + P23p + Z3p
    1.0, P23m + Z3m --> Z2m + P23m + Z3m
    1.0, P31p + Z1m --> Z3p + P31p + Z1m
    1.0, P31m + Z1p --> Z3p + P31m + Z1p
    1.0, P31p + Z1p --> Z3m + P31p + Z1p
    1.0, P31m + Z1m --> Z3m + P31m + Z1m
    1.0, P32p + Z2m --> Z3p + P32p + Z2m
    1.0, P32m + Z2p --> Z3p + P32m + Z2p
    1.0, P32p + Z2p --> Z3m + P32p + Z2p
    1.0, P32m + Z2m --> Z3m + P32m + Z2m
    1.0, P33p + Z3m --> Z3p + P33p + Z3m
    1.0, P33m + Z3p --> Z3p + P33m + Z3p
    1.0, P33p + Z3p --> Z3m + P33p + Z3p
    1.0, P33m + Z3m --> Z3m + P33m + Z3m
    1.0, A1p + P11p --> A1p + A1p + P11p
    1.0, A1m + P11m --> A1p + A1m + P11m
    1.0, A1p + P11m --> A1m + A1p + P11m
    1.0, A1m + P11p --> A1m + A1m + P11p
    1.0, A2p + P21p --> A1p + A2p + P21p
    1.0, A2m + P21m --> A1p + A2m + P21m
    1.0, A2p + P21m --> A1m + A2p + P21m
    1.0, A2m + P21p --> A1m + A2m + P21p
    1.0, A3p + P31p --> A1p + A3p + P31p
    1.0, A3m + P31m --> A1p + A3m + P31m
    1.0, A3p + P31m --> A1m + A3p + P31m
    1.0, A3m + P31p --> A1m + A3m + P31p
    1.0, A1p + P12p --> A2p + A1p + P12p
    1.0, A1m + P12m --> A2p + A1m + P12m
    1.0, A1p + P12m --> A2m + A1p + P12m
    1.0, A1m + P12p --> A2m + A1m + P12p
    1.0, A2p + P22p --> A2p + A2p + P22p
    1.0, A2m + P22m --> A2p + A2m + P22m
    1.0, A2p + P22m --> A2m + A2p + P22m
    1.0, A2m + P22p --> A2m + A2m + P22p
    1.0, A3p + P32p --> A2p + A3p + P32p
    1.0, A3m + P32m --> A2p + A3m + P32m
    1.0, A3p + P32m --> A2m + A3p + P32m
    1.0, A3m + P32p --> A2m + A3m + P32p
    1.0, A1p + P13p --> A3p + A1p + P13p
    1.0, A1m + P13m --> A3p + A1m + P13m
    1.0, A1p + P13m --> A3m + A1p + P13m
    1.0, A1m + P13p --> A3m + A1m + P13p
    1.0, A2p + P23p --> A3p + A2p + P23p
    1.0, A2m + P23m --> A3p + A2m + P23m
    1.0, A2p + P23m --> A3m + A2p + P23m
    1.0, A2m + P23p --> A3m + A2m + P23p
    1.0, A3p + P33p --> A3p + A3p + P33p
    1.0, A3m + P33m --> A3p + A3m + P33m
    1.0, A3p + P33m --> A3m + A3p + P33m
    1.0, A3m + P33p --> A3m + A3m + P33p
    1.0, A1p + Z1p --> G11p + A1p + Z1p
    1.0, A1m + Z1m --> G11p + A1m + Z1m
    1.0, A1p + Z1m --> G11m + A1p + Z1m
    1.0, A1m + Z1p --> G11m + A1m + Z1p
    1.0, A1p + Z2p --> G12p + A1p + Z2p
    1.0, A1m + Z2m --> G12p + A1m + Z2m
    1.0, A1p + Z2m --> G12m + A1p + Z2m
    1.0, A1m + Z2p --> G12m + A1m + Z2p
    1.0, A1p + Z3p --> G13p + A1p + Z3p
    1.0, A1m + Z3m --> G13p + A1m + Z3m
    1.0, A1p + Z3m --> G13m + A1p + Z3m
    1.0, A1m + Z3p --> G13m + A1m + Z3p
    1.0, A2p + Z1p --> G21p + A2p + Z1p
    1.0, A2m + Z1m --> G21p + A2m + Z1m
    1.0, A2p + Z1m --> G21m + A2p + Z1m
    1.0, A2m + Z1p --> G21m + A2m + Z1p
    1.0, A2p + Z2p --> G22p + A2p + Z2p
    1.0, A2m + Z2m --> G22p + A2m + Z2m
    1.0, A2p + Z2m --> G22m + A2p + Z2m
    1.0, A2m + Z2p --> G22m + A2m + Z2p
    1.0, A2p + Z3p --> G23p + A2p + Z3p
    1.0, A2m + Z3m --> G23p + A2m + Z3m
    1.0, A2p + Z3m --> G23m + A2p + Z3m
    1.0, A2m + Z3p --> G23m + A2m + Z3p
    1.0, A3p + Z1p --> G31p + A3p + Z1p
    1.0, A3m + Z1m --> G31p + A3m + Z1m
    1.0, A3p + Z1m --> G31m + A3p + Z1m
    1.0, A3m + Z1p --> G31m + A3m + Z1p
    1.0, A3p + Z2p --> G32p + A3p + Z2p
    1.0, A3m + Z2m --> G32p + A3m + Z2m
    1.0, A3p + Z2m --> G32m + A3p + Z2m
    1.0, A3m + Z2p --> G32m + A3m + Z2p
    1.0, A3p + Z3p --> G33p + A3p + Z3p
    1.0, A3m + Z3m --> G33p + A3m + Z3m
    1.0, A3p + Z3m --> G33m + A3p + Z3m
    1.0, A3m + Z3p --> G33m + A3m + Z3p
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
end k1 k2


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
end k1 k2


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
    1.0, Op + Ym --> Ep + Op + Ym
    1.0, Om + Yp --> Em + Om + Yp
end

rn_dual_binary_scalar_mult = @reaction_network rn_dual_binary_scalar_mult begin
    1.0, Ep + S1p --> P1p + Ep
    1.0, Ep + S1m --> P1m + Ep
    1.0, Ep + S2p --> P2p + Ep
    1.0, Ep + S2m --> P2m + Ep
    1.0, Ep + S3p --> P3p + Ep 
    1.0, Ep + S3m --> P3m + Ep 
    1.0, Em + S1p --> P1m + Em 
    1.0, Em + S1m --> P1p + Em 
    1.0, Em + S2p --> P2m + Em 
    1.0, Em + S2m --> P2p + Em 
    1.0, Em + S3p --> P3m + Em 
    1.0, Em + S3m --> P3p + Em 
end