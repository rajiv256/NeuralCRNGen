rn_ncrn_fwd = @reaction_network rn_ncrn_fwd begin
# dz_i/dt = p_i z_i
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
# dz_i/dt = -z^2
1.0, Z1p + Z1m --> Z1p + Z1p + Z1m
1.0, Z1m + Z1p --> Z1p + Z1m + Z1p
1.0, Z1p + Z1p --> Z1m + Z1p + Z1p
1.0, Z1m + Z1m --> Z1m + Z1m + Z1m
1.0, Z2p + Z2m --> Z2p + Z2p + Z2m
1.0, Z2m + Z2p --> Z2p + Z2m + Z2p
1.0, Z2p + Z2p --> Z2m + Z2p + Z2p
1.0, Z2m + Z2m --> Z2m + Z2m + Z2m
1.0, Z3p + Z3m --> Z3p + Z3p + Z3m
1.0, Z3m + Z3p --> Z3p + Z3m + Z3p
1.0, Z3p + Z3p --> Z3m + Z3p + Z3p
1.0, Z3m + Z3m --> Z3m + Z3m + Z3m
end
rn_ncrn_bwd = @reaction_network rn_ncrn_bwd begin
# dz/dt = -p_i z_i
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
1.0, Z1p + Z1p --> Z1p + Z1p + Z1p
1.0, Z1m + Z1m --> Z1p + Z1m + Z1m
1.0, Z1p + Z1m --> Z1m + Z1p + Z1m
1.0, Z1m + Z1p --> Z1m + Z1m + Z1p
1.0, Z2p + Z2p --> Z2p + Z2p + Z2p
1.0, Z2m + Z2m --> Z2p + Z2m + Z2m
1.0, Z2p + Z2m --> Z2m + Z2p + Z2m
1.0, Z2m + Z2p --> Z2m + Z2m + Z2p
1.0, Z3p + Z3p --> Z3p + Z3p + Z3p
1.0, Z3m + Z3m --> Z3p + Z3m + Z3m
1.0, Z3p + Z3m --> Z3m + Z3p + Z3m
1.0, Z3m + Z3p --> Z3m + Z3m + Z3p

# da_i/dt = a_i p_i
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

# da_i/dt = -a_i z_i
1.0, A1p + Z1m --> A1p + A1p + Z1m
1.0, A1m + Z1p --> A1p + A1m + Z1p
1.0, A1p + Z1p --> A1m + A1p + Z1p
1.0, A1m + Z1m --> A1m + A1m + Z1m
1.0, A2p + Z2m --> A2p + A2p + Z2m
1.0, A2m + Z2p --> A2p + A2m + Z2p
1.0, A2p + Z2p --> A2m + A2p + Z2p
1.0, A2m + Z2m --> A2m + A2m + Z2m
1.0, A3p + Z3m --> A3p + A3p + Z3m
1.0, A3m + Z3p --> A3p + A3m + Z3p
1.0, A3p + Z3p --> A3m + A3p + Z3p
1.0, A3m + Z3m --> A3m + A3m + Z3m
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
