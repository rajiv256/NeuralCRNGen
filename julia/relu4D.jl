rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
# dz_i/dt = h
1.0, H1p --> Z1p + H1p
1.0, H1m --> Z1m + H1m
1.0, H2p --> Z2p + H2p
1.0, H2m --> Z2m + H2m
1.0, H3p --> Z3p + H3p
1.0, H3m --> Z3m + H3m
1.0, H4p --> Z4p + H4p
1.0, H4m --> Z4m + H4m
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
1.0, P13p + Z1p + X3p --> Z1p + P13p + Z1p + X3p
1.0, P13m + Z1m + X3p --> Z1p + P13m + Z1m + X3p
1.0, P13p + Z1m + X3m --> Z1p + P13p + Z1m + X3m
1.0, P13m + Z1p + X3m --> Z1p + P13m + Z1p + X3m
1.0, P13p + Z1p + X3m --> Z1m + P13p + Z1p + X3m
1.0, P13m + Z1m + X3m --> Z1m + P13m + Z1m + X3m
1.0, P13p + Z1m + X3p --> Z1m + P13p + Z1m + X3p
1.0, P13m + Z1p + X3p --> Z1m + P13m + Z1p + X3p
1.0, P14p + Z1p + X4p --> Z1p + P14p + Z1p + X4p
1.0, P14m + Z1m + X4p --> Z1p + P14m + Z1m + X4p
1.0, P14p + Z1m + X4m --> Z1p + P14p + Z1m + X4m
1.0, P14m + Z1p + X4m --> Z1p + P14m + Z1p + X4m
1.0, P14p + Z1p + X4m --> Z1m + P14p + Z1p + X4m
1.0, P14m + Z1m + X4m --> Z1m + P14m + Z1m + X4m
1.0, P14p + Z1m + X4p --> Z1m + P14p + Z1m + X4p
1.0, P14m + Z1p + X4p --> Z1m + P14m + Z1p + X4p
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
1.0, P23p + Z2p + X3p --> Z2p + P23p + Z2p + X3p
1.0, P23m + Z2m + X3p --> Z2p + P23m + Z2m + X3p
1.0, P23p + Z2m + X3m --> Z2p + P23p + Z2m + X3m
1.0, P23m + Z2p + X3m --> Z2p + P23m + Z2p + X3m
1.0, P23p + Z2p + X3m --> Z2m + P23p + Z2p + X3m
1.0, P23m + Z2m + X3m --> Z2m + P23m + Z2m + X3m
1.0, P23p + Z2m + X3p --> Z2m + P23p + Z2m + X3p
1.0, P23m + Z2p + X3p --> Z2m + P23m + Z2p + X3p
1.0, P24p + Z2p + X4p --> Z2p + P24p + Z2p + X4p
1.0, P24m + Z2m + X4p --> Z2p + P24m + Z2m + X4p
1.0, P24p + Z2m + X4m --> Z2p + P24p + Z2m + X4m
1.0, P24m + Z2p + X4m --> Z2p + P24m + Z2p + X4m
1.0, P24p + Z2p + X4m --> Z2m + P24p + Z2p + X4m
1.0, P24m + Z2m + X4m --> Z2m + P24m + Z2m + X4m
1.0, P24p + Z2m + X4p --> Z2m + P24p + Z2m + X4p
1.0, P24m + Z2p + X4p --> Z2m + P24m + Z2p + X4p
1.0, P31p + Z3p + X1p --> Z3p + P31p + Z3p + X1p
1.0, P31m + Z3m + X1p --> Z3p + P31m + Z3m + X1p
1.0, P31p + Z3m + X1m --> Z3p + P31p + Z3m + X1m
1.0, P31m + Z3p + X1m --> Z3p + P31m + Z3p + X1m
1.0, P31p + Z3p + X1m --> Z3m + P31p + Z3p + X1m
1.0, P31m + Z3m + X1m --> Z3m + P31m + Z3m + X1m
1.0, P31p + Z3m + X1p --> Z3m + P31p + Z3m + X1p
1.0, P31m + Z3p + X1p --> Z3m + P31m + Z3p + X1p
1.0, P32p + Z3p + X2p --> Z3p + P32p + Z3p + X2p
1.0, P32m + Z3m + X2p --> Z3p + P32m + Z3m + X2p
1.0, P32p + Z3m + X2m --> Z3p + P32p + Z3m + X2m
1.0, P32m + Z3p + X2m --> Z3p + P32m + Z3p + X2m
1.0, P32p + Z3p + X2m --> Z3m + P32p + Z3p + X2m
1.0, P32m + Z3m + X2m --> Z3m + P32m + Z3m + X2m
1.0, P32p + Z3m + X2p --> Z3m + P32p + Z3m + X2p
1.0, P32m + Z3p + X2p --> Z3m + P32m + Z3p + X2p
1.0, P33p + Z3p + X3p --> Z3p + P33p + Z3p + X3p
1.0, P33m + Z3m + X3p --> Z3p + P33m + Z3m + X3p
1.0, P33p + Z3m + X3m --> Z3p + P33p + Z3m + X3m
1.0, P33m + Z3p + X3m --> Z3p + P33m + Z3p + X3m
1.0, P33p + Z3p + X3m --> Z3m + P33p + Z3p + X3m
1.0, P33m + Z3m + X3m --> Z3m + P33m + Z3m + X3m
1.0, P33p + Z3m + X3p --> Z3m + P33p + Z3m + X3p
1.0, P33m + Z3p + X3p --> Z3m + P33m + Z3p + X3p
1.0, P34p + Z3p + X4p --> Z3p + P34p + Z3p + X4p
1.0, P34m + Z3m + X4p --> Z3p + P34m + Z3m + X4p
1.0, P34p + Z3m + X4m --> Z3p + P34p + Z3m + X4m
1.0, P34m + Z3p + X4m --> Z3p + P34m + Z3p + X4m
1.0, P34p + Z3p + X4m --> Z3m + P34p + Z3p + X4m
1.0, P34m + Z3m + X4m --> Z3m + P34m + Z3m + X4m
1.0, P34p + Z3m + X4p --> Z3m + P34p + Z3m + X4p
1.0, P34m + Z3p + X4p --> Z3m + P34m + Z3p + X4p
1.0, P41p + Z4p + X1p --> Z4p + P41p + Z4p + X1p
1.0, P41m + Z4m + X1p --> Z4p + P41m + Z4m + X1p
1.0, P41p + Z4m + X1m --> Z4p + P41p + Z4m + X1m
1.0, P41m + Z4p + X1m --> Z4p + P41m + Z4p + X1m
1.0, P41p + Z4p + X1m --> Z4m + P41p + Z4p + X1m
1.0, P41m + Z4m + X1m --> Z4m + P41m + Z4m + X1m
1.0, P41p + Z4m + X1p --> Z4m + P41p + Z4m + X1p
1.0, P41m + Z4p + X1p --> Z4m + P41m + Z4p + X1p
1.0, P42p + Z4p + X2p --> Z4p + P42p + Z4p + X2p
1.0, P42m + Z4m + X2p --> Z4p + P42m + Z4m + X2p
1.0, P42p + Z4m + X2m --> Z4p + P42p + Z4m + X2m
1.0, P42m + Z4p + X2m --> Z4p + P42m + Z4p + X2m
1.0, P42p + Z4p + X2m --> Z4m + P42p + Z4p + X2m
1.0, P42m + Z4m + X2m --> Z4m + P42m + Z4m + X2m
1.0, P42p + Z4m + X2p --> Z4m + P42p + Z4m + X2p
1.0, P42m + Z4p + X2p --> Z4m + P42m + Z4p + X2p
1.0, P43p + Z4p + X3p --> Z4p + P43p + Z4p + X3p
1.0, P43m + Z4m + X3p --> Z4p + P43m + Z4m + X3p
1.0, P43p + Z4m + X3m --> Z4p + P43p + Z4m + X3m
1.0, P43m + Z4p + X3m --> Z4p + P43m + Z4p + X3m
1.0, P43p + Z4p + X3m --> Z4m + P43p + Z4p + X3m
1.0, P43m + Z4m + X3m --> Z4m + P43m + Z4m + X3m
1.0, P43p + Z4m + X3p --> Z4m + P43p + Z4m + X3p
1.0, P43m + Z4p + X3p --> Z4m + P43m + Z4p + X3p
1.0, P44p + Z4p + X4p --> Z4p + P44p + Z4p + X4p
1.0, P44m + Z4m + X4p --> Z4p + P44m + Z4m + X4p
1.0, P44p + Z4m + X4m --> Z4p + P44p + Z4m + X4m
1.0, P44m + Z4p + X4m --> Z4p + P44m + Z4p + X4m
1.0, P44p + Z4p + X4m --> Z4m + P44p + Z4p + X4m
1.0, P44m + Z4m + X4m --> Z4m + P44m + Z4m + X4m
1.0, P44p + Z4m + X4p --> Z4m + P44p + Z4m + X4p
1.0, P44m + Z4p + X4p --> Z4m + P44m + Z4p + X4p
# dz_i/dt = b_iz_i
1.0, B1p + Z1p --> Z1p + B1p + Z1p
1.0, B1m + Z1m --> Z1p + B1m + Z1m
1.0, B1p + Z1m --> Z1m + B1p + Z1m
1.0, B1m + Z1p --> Z1m + B1m + Z1p
1.0, B2p + Z2p --> Z2p + B2p + Z2p
1.0, B2m + Z2m --> Z2p + B2m + Z2m
1.0, B2p + Z2m --> Z2m + B2p + Z2m
1.0, B2m + Z2p --> Z2m + B2m + Z2p
1.0, B3p + Z3p --> Z3p + B3p + Z3p
1.0, B3m + Z3m --> Z3p + B3m + Z3m
1.0, B3p + Z3m --> Z3m + B3p + Z3m
1.0, B3m + Z3p --> Z3m + B3m + Z3p
1.0, B4p + Z4p --> Z4p + B4p + Z4p
1.0, B4m + Z4m --> Z4p + B4m + Z4m
1.0, B4p + Z4m --> Z4m + B4p + Z4m
1.0, B4m + Z4p --> Z4m + B4m + Z4p
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

1.0, Z4p + Z4p --> Z4p
1.0, Z4m + Z4m --> Z4m
100.0, Z4p + Z4m --> 0
end

rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin
# dz/dt = -h
1.0, H1m --> Z1p + H1m
1.0, H1p --> Z1m + H1p
1.0, H2m --> Z2p + H2m
1.0, H2p --> Z2m + H2p
1.0, H3m --> Z3p + H3m
1.0, H3p --> Z3m + H3p
1.0, H4m --> Z4p + H4m
1.0, H4p --> Z4m + H4p
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
1.0, P13p + Z1p + X3m --> Z1p + P13p + Z1p + X3m
1.0, P13m + Z1m + X3m --> Z1p + P13m + Z1m + X3m
1.0, P13p + Z1m + X3p --> Z1p + P13p + Z1m + X3p
1.0, P13m + Z1p + X3p --> Z1p + P13m + Z1p + X3p
1.0, P13p + Z1p + X3p --> Z1m + P13p + Z1p + X3p
1.0, P13m + Z1m + X3p --> Z1m + P13m + Z1m + X3p
1.0, P13p + Z1m + X3m --> Z1m + P13p + Z1m + X3m
1.0, P13m + Z1p + X3m --> Z1m + P13m + Z1p + X3m
1.0, P14p + Z1p + X4m --> Z1p + P14p + Z1p + X4m
1.0, P14m + Z1m + X4m --> Z1p + P14m + Z1m + X4m
1.0, P14p + Z1m + X4p --> Z1p + P14p + Z1m + X4p
1.0, P14m + Z1p + X4p --> Z1p + P14m + Z1p + X4p
1.0, P14p + Z1p + X4p --> Z1m + P14p + Z1p + X4p
1.0, P14m + Z1m + X4p --> Z1m + P14m + Z1m + X4p
1.0, P14p + Z1m + X4m --> Z1m + P14p + Z1m + X4m
1.0, P14m + Z1p + X4m --> Z1m + P14m + Z1p + X4m
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
1.0, P23p + Z2p + X3m --> Z2p + P23p + Z2p + X3m
1.0, P23m + Z2m + X3m --> Z2p + P23m + Z2m + X3m
1.0, P23p + Z2m + X3p --> Z2p + P23p + Z2m + X3p
1.0, P23m + Z2p + X3p --> Z2p + P23m + Z2p + X3p
1.0, P23p + Z2p + X3p --> Z2m + P23p + Z2p + X3p
1.0, P23m + Z2m + X3p --> Z2m + P23m + Z2m + X3p
1.0, P23p + Z2m + X3m --> Z2m + P23p + Z2m + X3m
1.0, P23m + Z2p + X3m --> Z2m + P23m + Z2p + X3m
1.0, P24p + Z2p + X4m --> Z2p + P24p + Z2p + X4m
1.0, P24m + Z2m + X4m --> Z2p + P24m + Z2m + X4m
1.0, P24p + Z2m + X4p --> Z2p + P24p + Z2m + X4p
1.0, P24m + Z2p + X4p --> Z2p + P24m + Z2p + X4p
1.0, P24p + Z2p + X4p --> Z2m + P24p + Z2p + X4p
1.0, P24m + Z2m + X4p --> Z2m + P24m + Z2m + X4p
1.0, P24p + Z2m + X4m --> Z2m + P24p + Z2m + X4m
1.0, P24m + Z2p + X4m --> Z2m + P24m + Z2p + X4m
1.0, P31p + Z3p + X1m --> Z3p + P31p + Z3p + X1m
1.0, P31m + Z3m + X1m --> Z3p + P31m + Z3m + X1m
1.0, P31p + Z3m + X1p --> Z3p + P31p + Z3m + X1p
1.0, P31m + Z3p + X1p --> Z3p + P31m + Z3p + X1p
1.0, P31p + Z3p + X1p --> Z3m + P31p + Z3p + X1p
1.0, P31m + Z3m + X1p --> Z3m + P31m + Z3m + X1p
1.0, P31p + Z3m + X1m --> Z3m + P31p + Z3m + X1m
1.0, P31m + Z3p + X1m --> Z3m + P31m + Z3p + X1m
1.0, P32p + Z3p + X2m --> Z3p + P32p + Z3p + X2m
1.0, P32m + Z3m + X2m --> Z3p + P32m + Z3m + X2m
1.0, P32p + Z3m + X2p --> Z3p + P32p + Z3m + X2p
1.0, P32m + Z3p + X2p --> Z3p + P32m + Z3p + X2p
1.0, P32p + Z3p + X2p --> Z3m + P32p + Z3p + X2p
1.0, P32m + Z3m + X2p --> Z3m + P32m + Z3m + X2p
1.0, P32p + Z3m + X2m --> Z3m + P32p + Z3m + X2m
1.0, P32m + Z3p + X2m --> Z3m + P32m + Z3p + X2m
1.0, P33p + Z3p + X3m --> Z3p + P33p + Z3p + X3m
1.0, P33m + Z3m + X3m --> Z3p + P33m + Z3m + X3m
1.0, P33p + Z3m + X3p --> Z3p + P33p + Z3m + X3p
1.0, P33m + Z3p + X3p --> Z3p + P33m + Z3p + X3p
1.0, P33p + Z3p + X3p --> Z3m + P33p + Z3p + X3p
1.0, P33m + Z3m + X3p --> Z3m + P33m + Z3m + X3p
1.0, P33p + Z3m + X3m --> Z3m + P33p + Z3m + X3m
1.0, P33m + Z3p + X3m --> Z3m + P33m + Z3p + X3m
1.0, P34p + Z3p + X4m --> Z3p + P34p + Z3p + X4m
1.0, P34m + Z3m + X4m --> Z3p + P34m + Z3m + X4m
1.0, P34p + Z3m + X4p --> Z3p + P34p + Z3m + X4p
1.0, P34m + Z3p + X4p --> Z3p + P34m + Z3p + X4p
1.0, P34p + Z3p + X4p --> Z3m + P34p + Z3p + X4p
1.0, P34m + Z3m + X4p --> Z3m + P34m + Z3m + X4p
1.0, P34p + Z3m + X4m --> Z3m + P34p + Z3m + X4m
1.0, P34m + Z3p + X4m --> Z3m + P34m + Z3p + X4m
1.0, P41p + Z4p + X1m --> Z4p + P41p + Z4p + X1m
1.0, P41m + Z4m + X1m --> Z4p + P41m + Z4m + X1m
1.0, P41p + Z4m + X1p --> Z4p + P41p + Z4m + X1p
1.0, P41m + Z4p + X1p --> Z4p + P41m + Z4p + X1p
1.0, P41p + Z4p + X1p --> Z4m + P41p + Z4p + X1p
1.0, P41m + Z4m + X1p --> Z4m + P41m + Z4m + X1p
1.0, P41p + Z4m + X1m --> Z4m + P41p + Z4m + X1m
1.0, P41m + Z4p + X1m --> Z4m + P41m + Z4p + X1m
1.0, P42p + Z4p + X2m --> Z4p + P42p + Z4p + X2m
1.0, P42m + Z4m + X2m --> Z4p + P42m + Z4m + X2m
1.0, P42p + Z4m + X2p --> Z4p + P42p + Z4m + X2p
1.0, P42m + Z4p + X2p --> Z4p + P42m + Z4p + X2p
1.0, P42p + Z4p + X2p --> Z4m + P42p + Z4p + X2p
1.0, P42m + Z4m + X2p --> Z4m + P42m + Z4m + X2p
1.0, P42p + Z4m + X2m --> Z4m + P42p + Z4m + X2m
1.0, P42m + Z4p + X2m --> Z4m + P42m + Z4p + X2m
1.0, P43p + Z4p + X3m --> Z4p + P43p + Z4p + X3m
1.0, P43m + Z4m + X3m --> Z4p + P43m + Z4m + X3m
1.0, P43p + Z4m + X3p --> Z4p + P43p + Z4m + X3p
1.0, P43m + Z4p + X3p --> Z4p + P43m + Z4p + X3p
1.0, P43p + Z4p + X3p --> Z4m + P43p + Z4p + X3p
1.0, P43m + Z4m + X3p --> Z4m + P43m + Z4m + X3p
1.0, P43p + Z4m + X3m --> Z4m + P43p + Z4m + X3m
1.0, P43m + Z4p + X3m --> Z4m + P43m + Z4p + X3m
1.0, P44p + Z4p + X4m --> Z4p + P44p + Z4p + X4m
1.0, P44m + Z4m + X4m --> Z4p + P44m + Z4m + X4m
1.0, P44p + Z4m + X4p --> Z4p + P44p + Z4m + X4p
1.0, P44m + Z4p + X4p --> Z4p + P44m + Z4p + X4p
1.0, P44p + Z4p + X4p --> Z4m + P44p + Z4p + X4p
1.0, P44m + Z4m + X4p --> Z4m + P44m + Z4m + X4p
1.0, P44p + Z4m + X4m --> Z4m + P44p + Z4m + X4m
1.0, P44m + Z4p + X4m --> Z4m + P44m + Z4p + X4m
# dz/dt = -b_i z_i
1.0, B1p + Z1m --> Z1p + B1p + Z1m
1.0, B1m + Z1p --> Z1p + B1m + Z1p
1.0, B1p + Z1p --> Z1m + B1p + Z1p
1.0, B1m + Z1m --> Z1m + B1m + Z1m
1.0, B2p + Z2m --> Z2p + B2p + Z2m
1.0, B2m + Z2p --> Z2p + B2m + Z2p
1.0, B2p + Z2p --> Z2m + B2p + Z2p
1.0, B2m + Z2m --> Z2m + B2m + Z2m
1.0, B3p + Z3m --> Z3p + B3p + Z3m
1.0, B3m + Z3p --> Z3p + B3m + Z3p
1.0, B3p + Z3p --> Z3m + B3p + Z3p
1.0, B3m + Z3m --> Z3m + B3m + Z3m
1.0, B4p + Z4m --> Z4p + B4p + Z4m
1.0, B4m + Z4p --> Z4p + B4m + Z4p
1.0, B4p + Z4p --> Z4m + B4p + Z4p
1.0, B4m + Z4m --> Z4m + B4m + Z4m
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

10.0, Z4p + Z4m --> 0
1.0, Z4p + Z4p --> Z4p + Z4p + Z4p
1.0, Z4m + Z4m --> Z4m + Z4m + Z4m

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
    1.0, A1p + P13p + X3p --> A1p + A1p + P13p + X3p
    1.0, A1m + P13m + X3p --> A1p + A1m + P13m + X3p
    1.0, A1p + P13m + X3m --> A1p + A1p + P13m + X3m
    1.0, A1m + P13p + X3m --> A1p + A1m + P13p + X3m
    1.0, A1p + P13p + X3m --> A1m + A1p + P13p + X3m
    1.0, A1m + P13m + X3m --> A1m + A1m + P13m + X3m
    1.0, A1p + P13m + X3p --> A1m + A1p + P13m + X3p
    1.0, A1m + P13p + X3p --> A1m + A1m + P13p + X3p
    1.0, A1p + P14p + X4p --> A1p + A1p + P14p + X4p
    1.0, A1m + P14m + X4p --> A1p + A1m + P14m + X4p
    1.0, A1p + P14m + X4m --> A1p + A1p + P14m + X4m
    1.0, A1m + P14p + X4m --> A1p + A1m + P14p + X4m
    1.0, A1p + P14p + X4m --> A1m + A1p + P14p + X4m
    1.0, A1m + P14m + X4m --> A1m + A1m + P14m + X4m
    1.0, A1p + P14m + X4p --> A1m + A1p + P14m + X4p
    1.0, A1m + P14p + X4p --> A1m + A1m + P14p + X4p
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
    1.0, A2p + P23p + X3p --> A2p + A2p + P23p + X3p
    1.0, A2m + P23m + X3p --> A2p + A2m + P23m + X3p
    1.0, A2p + P23m + X3m --> A2p + A2p + P23m + X3m
    1.0, A2m + P23p + X3m --> A2p + A2m + P23p + X3m
    1.0, A2p + P23p + X3m --> A2m + A2p + P23p + X3m
    1.0, A2m + P23m + X3m --> A2m + A2m + P23m + X3m
    1.0, A2p + P23m + X3p --> A2m + A2p + P23m + X3p
    1.0, A2m + P23p + X3p --> A2m + A2m + P23p + X3p
    1.0, A2p + P24p + X4p --> A2p + A2p + P24p + X4p
    1.0, A2m + P24m + X4p --> A2p + A2m + P24m + X4p
    1.0, A2p + P24m + X4m --> A2p + A2p + P24m + X4m
    1.0, A2m + P24p + X4m --> A2p + A2m + P24p + X4m
    1.0, A2p + P24p + X4m --> A2m + A2p + P24p + X4m
    1.0, A2m + P24m + X4m --> A2m + A2m + P24m + X4m
    1.0, A2p + P24m + X4p --> A2m + A2p + P24m + X4p
    1.0, A2m + P24p + X4p --> A2m + A2m + P24p + X4p
    1.0, A3p + P31p + X1p --> A3p + A3p + P31p + X1p
    1.0, A3m + P31m + X1p --> A3p + A3m + P31m + X1p
    1.0, A3p + P31m + X1m --> A3p + A3p + P31m + X1m
    1.0, A3m + P31p + X1m --> A3p + A3m + P31p + X1m
    1.0, A3p + P31p + X1m --> A3m + A3p + P31p + X1m
    1.0, A3m + P31m + X1m --> A3m + A3m + P31m + X1m
    1.0, A3p + P31m + X1p --> A3m + A3p + P31m + X1p
    1.0, A3m + P31p + X1p --> A3m + A3m + P31p + X1p
    1.0, A3p + P32p + X2p --> A3p + A3p + P32p + X2p
    1.0, A3m + P32m + X2p --> A3p + A3m + P32m + X2p
    1.0, A3p + P32m + X2m --> A3p + A3p + P32m + X2m
    1.0, A3m + P32p + X2m --> A3p + A3m + P32p + X2m
    1.0, A3p + P32p + X2m --> A3m + A3p + P32p + X2m
    1.0, A3m + P32m + X2m --> A3m + A3m + P32m + X2m
    1.0, A3p + P32m + X2p --> A3m + A3p + P32m + X2p
    1.0, A3m + P32p + X2p --> A3m + A3m + P32p + X2p
    1.0, A3p + P33p + X3p --> A3p + A3p + P33p + X3p
    1.0, A3m + P33m + X3p --> A3p + A3m + P33m + X3p
    1.0, A3p + P33m + X3m --> A3p + A3p + P33m + X3m
    1.0, A3m + P33p + X3m --> A3p + A3m + P33p + X3m
    1.0, A3p + P33p + X3m --> A3m + A3p + P33p + X3m
    1.0, A3m + P33m + X3m --> A3m + A3m + P33m + X3m
    1.0, A3p + P33m + X3p --> A3m + A3p + P33m + X3p
    1.0, A3m + P33p + X3p --> A3m + A3m + P33p + X3p
    1.0, A3p + P34p + X4p --> A3p + A3p + P34p + X4p
    1.0, A3m + P34m + X4p --> A3p + A3m + P34m + X4p
    1.0, A3p + P34m + X4m --> A3p + A3p + P34m + X4m
    1.0, A3m + P34p + X4m --> A3p + A3m + P34p + X4m
    1.0, A3p + P34p + X4m --> A3m + A3p + P34p + X4m
    1.0, A3m + P34m + X4m --> A3m + A3m + P34m + X4m
    1.0, A3p + P34m + X4p --> A3m + A3p + P34m + X4p
    1.0, A3m + P34p + X4p --> A3m + A3m + P34p + X4p
    1.0, A4p + P41p + X1p --> A4p + A4p + P41p + X1p
    1.0, A4m + P41m + X1p --> A4p + A4m + P41m + X1p
    1.0, A4p + P41m + X1m --> A4p + A4p + P41m + X1m
    1.0, A4m + P41p + X1m --> A4p + A4m + P41p + X1m
    1.0, A4p + P41p + X1m --> A4m + A4p + P41p + X1m
    1.0, A4m + P41m + X1m --> A4m + A4m + P41m + X1m
    1.0, A4p + P41m + X1p --> A4m + A4p + P41m + X1p
    1.0, A4m + P41p + X1p --> A4m + A4m + P41p + X1p
    1.0, A4p + P42p + X2p --> A4p + A4p + P42p + X2p
    1.0, A4m + P42m + X2p --> A4p + A4m + P42m + X2p
    1.0, A4p + P42m + X2m --> A4p + A4p + P42m + X2m
    1.0, A4m + P42p + X2m --> A4p + A4m + P42p + X2m
    1.0, A4p + P42p + X2m --> A4m + A4p + P42p + X2m
    1.0, A4m + P42m + X2m --> A4m + A4m + P42m + X2m
    1.0, A4p + P42m + X2p --> A4m + A4p + P42m + X2p
    1.0, A4m + P42p + X2p --> A4m + A4m + P42p + X2p
    1.0, A4p + P43p + X3p --> A4p + A4p + P43p + X3p
    1.0, A4m + P43m + X3p --> A4p + A4m + P43m + X3p
    1.0, A4p + P43m + X3m --> A4p + A4p + P43m + X3m
    1.0, A4m + P43p + X3m --> A4p + A4m + P43p + X3m
    1.0, A4p + P43p + X3m --> A4m + A4p + P43p + X3m
    1.0, A4m + P43m + X3m --> A4m + A4m + P43m + X3m
    1.0, A4p + P43m + X3p --> A4m + A4p + P43m + X3p
    1.0, A4m + P43p + X3p --> A4m + A4m + P43p + X3p
    1.0, A4p + P44p + X4p --> A4p + A4p + P44p + X4p
    1.0, A4m + P44m + X4p --> A4p + A4m + P44m + X4p
    1.0, A4p + P44m + X4m --> A4p + A4p + P44m + X4m
    1.0, A4m + P44p + X4m --> A4p + A4m + P44p + X4m
    1.0, A4p + P44p + X4m --> A4m + A4p + P44p + X4m
    1.0, A4m + P44m + X4m --> A4m + A4m + P44m + X4m
    1.0, A4p + P44m + X4p --> A4m + A4p + P44m + X4p
    1.0, A4m + P44p + X4p --> A4m + A4m + P44p + X4p
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
1.0, A1p + Z1p + X3p --> G13p + A1p + Z1p + X3p
1.0, A1m + Z1m + X3p --> G13p + A1m + Z1m + X3p
1.0, A1p + Z1m + X3m --> G13p + A1p + Z1m + X3m
1.0, A1m + Z1p + X3m --> G13p + A1m + Z1p + X3m
1.0, A1p + Z1p + X3m --> G13m + A1p + Z1p + X3m
1.0, A1m + Z1m + X3m --> G13m + A1m + Z1m + X3m
1.0, A1p + Z1m + X3p --> G13m + A1p + Z1m + X3p
1.0, A1m + Z1p + X3p --> G13m + A1m + Z1p + X3p
1.0, A1p + Z1p + X4p --> G14p + A1p + Z1p + X4p
1.0, A1m + Z1m + X4p --> G14p + A1m + Z1m + X4p
1.0, A1p + Z1m + X4m --> G14p + A1p + Z1m + X4m
1.0, A1m + Z1p + X4m --> G14p + A1m + Z1p + X4m
1.0, A1p + Z1p + X4m --> G14m + A1p + Z1p + X4m
1.0, A1m + Z1m + X4m --> G14m + A1m + Z1m + X4m
1.0, A1p + Z1m + X4p --> G14m + A1p + Z1m + X4p
1.0, A1m + Z1p + X4p --> G14m + A1m + Z1p + X4p
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
1.0, A2p + Z2p + X3p --> G23p + A2p + Z2p + X3p
1.0, A2m + Z2m + X3p --> G23p + A2m + Z2m + X3p
1.0, A2p + Z2m + X3m --> G23p + A2p + Z2m + X3m
1.0, A2m + Z2p + X3m --> G23p + A2m + Z2p + X3m
1.0, A2p + Z2p + X3m --> G23m + A2p + Z2p + X3m
1.0, A2m + Z2m + X3m --> G23m + A2m + Z2m + X3m
1.0, A2p + Z2m + X3p --> G23m + A2p + Z2m + X3p
1.0, A2m + Z2p + X3p --> G23m + A2m + Z2p + X3p
1.0, A2p + Z2p + X4p --> G24p + A2p + Z2p + X4p
1.0, A2m + Z2m + X4p --> G24p + A2m + Z2m + X4p
1.0, A2p + Z2m + X4m --> G24p + A2p + Z2m + X4m
1.0, A2m + Z2p + X4m --> G24p + A2m + Z2p + X4m
1.0, A2p + Z2p + X4m --> G24m + A2p + Z2p + X4m
1.0, A2m + Z2m + X4m --> G24m + A2m + Z2m + X4m
1.0, A2p + Z2m + X4p --> G24m + A2p + Z2m + X4p
1.0, A2m + Z2p + X4p --> G24m + A2m + Z2p + X4p
1.0, A3p + Z3p + X1p --> G31p + A3p + Z3p + X1p
1.0, A3m + Z3m + X1p --> G31p + A3m + Z3m + X1p
1.0, A3p + Z3m + X1m --> G31p + A3p + Z3m + X1m
1.0, A3m + Z3p + X1m --> G31p + A3m + Z3p + X1m
1.0, A3p + Z3p + X1m --> G31m + A3p + Z3p + X1m
1.0, A3m + Z3m + X1m --> G31m + A3m + Z3m + X1m
1.0, A3p + Z3m + X1p --> G31m + A3p + Z3m + X1p
1.0, A3m + Z3p + X1p --> G31m + A3m + Z3p + X1p
1.0, A3p + Z3p + X2p --> G32p + A3p + Z3p + X2p
1.0, A3m + Z3m + X2p --> G32p + A3m + Z3m + X2p
1.0, A3p + Z3m + X2m --> G32p + A3p + Z3m + X2m
1.0, A3m + Z3p + X2m --> G32p + A3m + Z3p + X2m
1.0, A3p + Z3p + X2m --> G32m + A3p + Z3p + X2m
1.0, A3m + Z3m + X2m --> G32m + A3m + Z3m + X2m
1.0, A3p + Z3m + X2p --> G32m + A3p + Z3m + X2p
1.0, A3m + Z3p + X2p --> G32m + A3m + Z3p + X2p
1.0, A3p + Z3p + X3p --> G33p + A3p + Z3p + X3p
1.0, A3m + Z3m + X3p --> G33p + A3m + Z3m + X3p
1.0, A3p + Z3m + X3m --> G33p + A3p + Z3m + X3m
1.0, A3m + Z3p + X3m --> G33p + A3m + Z3p + X3m
1.0, A3p + Z3p + X3m --> G33m + A3p + Z3p + X3m
1.0, A3m + Z3m + X3m --> G33m + A3m + Z3m + X3m
1.0, A3p + Z3m + X3p --> G33m + A3p + Z3m + X3p
1.0, A3m + Z3p + X3p --> G33m + A3m + Z3p + X3p
1.0, A3p + Z3p + X4p --> G34p + A3p + Z3p + X4p
1.0, A3m + Z3m + X4p --> G34p + A3m + Z3m + X4p
1.0, A3p + Z3m + X4m --> G34p + A3p + Z3m + X4m
1.0, A3m + Z3p + X4m --> G34p + A3m + Z3p + X4m
1.0, A3p + Z3p + X4m --> G34m + A3p + Z3p + X4m
1.0, A3m + Z3m + X4m --> G34m + A3m + Z3m + X4m
1.0, A3p + Z3m + X4p --> G34m + A3p + Z3m + X4p
1.0, A3m + Z3p + X4p --> G34m + A3m + Z3p + X4p
1.0, A4p + Z4p + X1p --> G41p + A4p + Z4p + X1p
1.0, A4m + Z4m + X1p --> G41p + A4m + Z4m + X1p
1.0, A4p + Z4m + X1m --> G41p + A4p + Z4m + X1m
1.0, A4m + Z4p + X1m --> G41p + A4m + Z4p + X1m
1.0, A4p + Z4p + X1m --> G41m + A4p + Z4p + X1m
1.0, A4m + Z4m + X1m --> G41m + A4m + Z4m + X1m
1.0, A4p + Z4m + X1p --> G41m + A4p + Z4m + X1p
1.0, A4m + Z4p + X1p --> G41m + A4m + Z4p + X1p
1.0, A4p + Z4p + X2p --> G42p + A4p + Z4p + X2p
1.0, A4m + Z4m + X2p --> G42p + A4m + Z4m + X2p
1.0, A4p + Z4m + X2m --> G42p + A4p + Z4m + X2m
1.0, A4m + Z4p + X2m --> G42p + A4m + Z4p + X2m
1.0, A4p + Z4p + X2m --> G42m + A4p + Z4p + X2m
1.0, A4m + Z4m + X2m --> G42m + A4m + Z4m + X2m
1.0, A4p + Z4m + X2p --> G42m + A4p + Z4m + X2p
1.0, A4m + Z4p + X2p --> G42m + A4m + Z4p + X2p
1.0, A4p + Z4p + X3p --> G43p + A4p + Z4p + X3p
1.0, A4m + Z4m + X3p --> G43p + A4m + Z4m + X3p
1.0, A4p + Z4m + X3m --> G43p + A4p + Z4m + X3m
1.0, A4m + Z4p + X3m --> G43p + A4m + Z4p + X3m
1.0, A4p + Z4p + X3m --> G43m + A4p + Z4p + X3m
1.0, A4m + Z4m + X3m --> G43m + A4m + Z4m + X3m
1.0, A4p + Z4m + X3p --> G43m + A4p + Z4m + X3p
1.0, A4m + Z4p + X3p --> G43m + A4m + Z4p + X3p
1.0, A4p + Z4p + X4p --> G44p + A4p + Z4p + X4p
1.0, A4m + Z4m + X4p --> G44p + A4m + Z4m + X4p
1.0, A4p + Z4m + X4m --> G44p + A4p + Z4m + X4m
1.0, A4m + Z4p + X4m --> G44p + A4m + Z4p + X4m
1.0, A4p + Z4p + X4m --> G44m + A4p + Z4p + X4m
1.0, A4m + Z4m + X4m --> G44m + A4m + Z4m + X4m
1.0, A4p + Z4m + X4p --> G44m + A4p + Z4m + X4p
1.0, A4m + Z4p + X4p --> G44m + A4m + Z4p + X4p
# dbg_i/dt = a_i z_i
1.0, A1p + Z1p --> V1p + A1p + Z1p
1.0, A1m + Z1m --> V1p + A1m + Z1m
1.0, A1p + Z1m --> V1m + A1p + Z1m
1.0, A1m + Z1p --> V1m + A1m + Z1p
1.0, A2p + Z2p --> V2p + A2p + Z2p
1.0, A2m + Z2m --> V2p + A2m + Z2m
1.0, A2p + Z2m --> V2m + A2p + Z2m
1.0, A2m + Z2p --> V2m + A2m + Z2p
1.0, A3p + Z3p --> V3p + A3p + Z3p
1.0, A3m + Z3m --> V3p + A3m + Z3m
1.0, A3p + Z3m --> V3m + A3p + Z3m
1.0, A3m + Z3p --> V3m + A3m + Z3p
1.0, A4p + Z4p --> V4p + A4p + Z4p
1.0, A4m + Z4m --> V4p + A4m + Z4m
1.0, A4p + Z4m --> V4m + A4p + Z4m
1.0, A4m + Z4p --> V4m + A4m + Z4p
end

rn_dual_mult = @reaction_network rn_dual_mult begin
    1.0, J1p + K1p --> Lp + J1p + K1p
    1.0, J1m + K1m --> Lp + J1m + K1m
    1.0, J1p + K1m --> Lm + J1p + K1m
    1.0, J1m + K1p --> Lm + J1m + K1p
    1.0, J2p + K2p --> Lp + J2p + K2p
    1.0, J2m + K2m --> Lp + J2m + K2m
    1.0, J2p + K2m --> Lm + J2p + K2m
    1.0, J2m + K2p --> Lm + J2m + K2p
end


rn_param_update = @reaction_network rn_param_update begin
    k1, G11p --> P11m
    k1, G11m --> P11p
    k1, G12p --> P12m
    k1, G12m --> P12p
    k1, G13p --> P13m
    k1, G13m --> P13p
    k1, G14p --> P14m
    k1, G14m --> P14p
    k1, G21p --> P21m
    k1, G21m --> P21p
    k1, G22p --> P22m
    k1, G22m --> P22p
    k1, G23p --> P23m
    k1, G23m --> P23p
    k1, G24p --> P24m
    k1, G24m --> P24p
    k1, G31p --> P31m
    k1, G31m --> P31p
    k1, G32p --> P32m
    k1, G32m --> P32p
    k1, G33p --> P33m
    k1, G33m --> P33p
    k1, G34p --> P34m
    k1, G34m --> P34p
    k1, G41p --> P41m
    k1, G41m --> P41p
    k1, G42p --> P42m
    k1, G42m --> P42p
    k1, G43p --> P43m
    k1, G43m --> P43p
    k1, G44p --> P44m
    k1, G44m --> P44p
    k2, G11p --> 0
    k2, G11m --> 0
    k2, G12p --> 0
    k2, G12m --> 0
    k2, G13p --> 0
    k2, G13m --> 0
    k2, G14p --> 0
    k2, G14m --> 0
    k2, G21p --> 0
    k2, G21m --> 0
    k2, G22p --> 0
    k2, G22m --> 0
    k2, G23p --> 0
    k2, G23m --> 0
    k2, G24p --> 0
    k2, G24m --> 0
    k2, G31p --> 0
    k2, G31m --> 0
    k2, G32p --> 0
    k2, G32m --> 0
    k2, G33p --> 0
    k2, G33m --> 0
    k2, G34p --> 0
    k2, G34m --> 0
    k2, G41p --> 0
    k2, G41m --> 0
    k2, G42p --> 0
    k2, G42m --> 0
    k2, G43p --> 0
    k2, G43m --> 0
    k2, G44p --> 0
    k2, G44m --> 0
    k1, V1p --> B1m
    k1, V1m --> B1p
    k1, V2p --> B2m
    k1, V2m --> B2p
    k1, V3p --> B3m
    k1, V3m --> B3p
    k1, V4p --> B4m
    k1, V4m --> B4p
    k2, V1p --> 0
    k2, V1m --> 0
    k2, V2p --> 0
    k2, V2m --> 0
    k2, V3p --> 0
    k2, V3m --> 0
    k2, V4p --> 0
    k2, V4m --> 0
end


rn_final_layer_update = @reaction_network rn_final_layer_update begin
    k1, M1p --> W1m
    k1, M1m --> W1p
    k1, M2p --> W2m
    k1, M2m --> W2p
    k1, M3p --> W3m
    k1, M3m --> W3p
    k1, M4p --> W4m
    k1, M4m --> W4p
    k2, M1p --> 0
    k2, M1m --> 0
    k2, M2p --> 0
    k2, M2m --> 0
    k2, M3p --> 0
    k2, M3m --> 0
    k2, M4p --> 0
    k2, M4m --> 0
end


rn_dissipate_reactions = @reaction_network rn_dissipate_reactions begin
    1.0, G11p --> 0
    1.0, G11m --> 0
    1.0, G12p --> 0
    1.0, G12m --> 0
    1.0, G13p --> 0
    1.0, G13m --> 0
    1.0, G14p --> 0
    1.0, G14m --> 0
    1.0, G21p --> 0
    1.0, G21m --> 0
    1.0, G22p --> 0
    1.0, G22m --> 0
    1.0, G23p --> 0
    1.0, G23m --> 0
    1.0, G24p --> 0
    1.0, G24m --> 0
    1.0, G31p --> 0
    1.0, G31m --> 0
    1.0, G32p --> 0
    1.0, G32m --> 0
    1.0, G33p --> 0
    1.0, G33m --> 0
    1.0, G34p --> 0
    1.0, G34m --> 0
    1.0, G41p --> 0
    1.0, G41m --> 0
    1.0, G42p --> 0
    1.0, G42m --> 0
    1.0, G43p --> 0
    1.0, G43m --> 0
    1.0, G44p --> 0
    1.0, G44m --> 0
    1.0, M1p --> 0
    1.0, M1m --> 0
    1.0, M2p --> 0
    1.0, M2m --> 0
    1.0, M3p --> 0
    1.0, M3m --> 0
    1.0, M4p --> 0
    1.0, M4m --> 0
    1.0, A1p --> 0
    1.0, A1m --> 0
    1.0, A2p --> 0
    1.0, A2m --> 0
    1.0, A3p --> 0
    1.0, A3m --> 0
    1.0, A4p --> 0
    1.0, A4m --> 0
    1.0, Ep --> 0
    1.0, Em --> 0
    1.0, Op --> 0
    1.0, Om --> 0
    1.0, Yp --> 0
    1.0, Ym --> 0
end


rn_dual_mult = @reaction_network rn_dual_mult begin
    1.0, Jp + Kp --> Lp + Jp + Kp
    1.0, Jm + Km --> Lp + Jm + Km
    1.0, Jp + Km --> Lm + Jp + Km
    1.0, Jm + Kp --> Lm + Jm + Kp
    1.0, Lp --> 0
    1.0, Lm --> 0
end


rn_dual_dot = @reaction_network rn_dual_dot begin
    1.0, J1p + K1p --> Lp + J1p + K1p
    1.0, J1m + K1m --> Lp + J1m + K1m
    1.0, J2p + K2p --> Lp + J2p + K2p
    1.0, J2m + K2m --> Lp + J2m + K2m
    1.0, J3p + K3p --> Lp + J3p + K3p
    1.0, J3m + K3m --> Lp + J3m + K3m
    1.0, J4p + K4p --> Lp + J4p + K4p
    1.0, J4m + K4m --> Lp + J4m + K4m
    1.0, Lp --> 0
    1.0, J1p + K1m --> Lm + J1p + K1m
    1.0, J1m + K1p --> Lm + J1m + K1p
    1.0, J2p + K2m --> Lm + J2p + K2m
    1.0, J2m + K2p --> Lm + J2m + K2p
    1.0, J3p + K3m --> Lm + J3p + K3m
    1.0, J3m + K3p --> Lm + J3m + K3p
    1.0, J4p + K4m --> Lm + J4p + K4m
    1.0, J4m + K4p --> Lm + J4m + K4p
    1.0, Lm --> 0
end


rn_dual_subtract = @reaction_network rn_dual_subtract begin
    1.0, Jp --> Lp
    1.0, Jm --> Lm
    1.0, Kp --> Lm
    1.0, Km --> Lp
end


rn_dual_add = @reaction_network rn_dual_add begin
    1.0, Jp --> Lp
    1.0, Kp --> Lp
    1.0, Jm --> Lm
    1.0, Km --> Lm
end


rn_output_annihilation = @reaction_network rn_output_annihilation begin
    1.0, Op + Om --> 0
end


rn_create_error_species = @reaction_network rn_create_error_species begin
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
    1.0, Ep + S4p --> P4p + Ep + S4p
    1.0, Ep + S4m --> P4m + Ep + S4m

    1.0, Em + S1p --> P1m + Em + S1p
    1.0, Em + S1m --> P1p + Em + S1m
    1.0, Em + S2p --> P2m + Em + S2p
    1.0, Em + S2m --> P2p + Em + S2m
    1.0, Em + S3p --> P3m + Em + S3p
    1.0, Em + S3m --> P3p + Em + S3m
    1.0, Em + S4p --> P4m + Em + S4p
    1.0, Em + S4m --> P4p + Em + S4m
    
    1.0, P1p --> 0
    1.0, P1m --> 0
    1.0, P2p --> 0
    1.0, P2m --> 0
    1.0, P3p --> 0
    1.0, P3m --> 0
    1.0, P4p --> 0
    1.0, P4m --> 0
end