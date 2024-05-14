rn_dual_node_relu_fwd = @reaction_network rn_dual_node_relu_fwd begin
    # dz_i/dt = h
    1.0, H0p --> Z0p + H0p
    1.0, H0m --> Z0m + H0m
    1.0, H1p --> Z1p + H1p
    1.0, H1m --> Z1m + H1m
    1.0, H2p --> Z2p + H2p
    1.0, H2m --> Z2m + H2m
    1.0, H3p --> Z3p + H3p
    1.0, H3m --> Z3m + H3m
    1.0, H4p --> Z4p + H4p
    1.0, H4m --> Z4m + H4m
    # dz_i/dt = p_ij x_j z_i
    1.0, P00p + Z0p + X0p --> Z0p + P00p + Z0p + X0p
    1.0, P00m + Z0m + X0p --> Z0p + P00m + Z0m + X0p
    1.0, P00p + Z0m + X0m --> Z0p + P00p + Z0m + X0m
    1.0, P00m + Z0p + X0m --> Z0p + P00m + Z0p + X0m
    1.0, P00p + Z0p + X0m --> Z0m + P00p + Z0p + X0m
    1.0, P00m + Z0m + X0m --> Z0m + P00m + Z0m + X0m
    1.0, P00p + Z0m + X0p --> Z0m + P00p + Z0m + X0p
    1.0, P00m + Z0p + X0p --> Z0m + P00m + Z0p + X0p
    1.0, P01p + Z0p + X1p --> Z0p + P01p + Z0p + X1p
    1.0, P01m + Z0m + X1p --> Z0p + P01m + Z0m + X1p
    1.0, P01p + Z0m + X1m --> Z0p + P01p + Z0m + X1m
    1.0, P01m + Z0p + X1m --> Z0p + P01m + Z0p + X1m
    1.0, P01p + Z0p + X1m --> Z0m + P01p + Z0p + X1m
    1.0, P01m + Z0m + X1m --> Z0m + P01m + Z0m + X1m
    1.0, P01p + Z0m + X1p --> Z0m + P01p + Z0m + X1p
    1.0, P01m + Z0p + X1p --> Z0m + P01m + Z0p + X1p
    1.0, P02p + Z0p + X2p --> Z0p + P02p + Z0p + X2p
    1.0, P02m + Z0m + X2p --> Z0p + P02m + Z0m + X2p
    1.0, P02p + Z0m + X2m --> Z0p + P02p + Z0m + X2m
    1.0, P02m + Z0p + X2m --> Z0p + P02m + Z0p + X2m
    1.0, P02p + Z0p + X2m --> Z0m + P02p + Z0p + X2m
    1.0, P02m + Z0m + X2m --> Z0m + P02m + Z0m + X2m
    1.0, P02p + Z0m + X2p --> Z0m + P02p + Z0m + X2p
    1.0, P02m + Z0p + X2p --> Z0m + P02m + Z0p + X2p
    1.0, P03p + Z0p + X3p --> Z0p + P03p + Z0p + X3p
    1.0, P03m + Z0m + X3p --> Z0p + P03m + Z0m + X3p
    1.0, P03p + Z0m + X3m --> Z0p + P03p + Z0m + X3m
    1.0, P03m + Z0p + X3m --> Z0p + P03m + Z0p + X3m
    1.0, P03p + Z0p + X3m --> Z0m + P03p + Z0p + X3m
    1.0, P03m + Z0m + X3m --> Z0m + P03m + Z0m + X3m
    1.0, P03p + Z0m + X3p --> Z0m + P03p + Z0m + X3p
    1.0, P03m + Z0p + X3p --> Z0m + P03m + Z0p + X3p
    1.0, P04p + Z0p + X4p --> Z0p + P04p + Z0p + X4p
    1.0, P04m + Z0m + X4p --> Z0p + P04m + Z0m + X4p
    1.0, P04p + Z0m + X4m --> Z0p + P04p + Z0m + X4m
    1.0, P04m + Z0p + X4m --> Z0p + P04m + Z0p + X4m
    1.0, P04p + Z0p + X4m --> Z0m + P04p + Z0p + X4m
    1.0, P04m + Z0m + X4m --> Z0m + P04m + Z0m + X4m
    1.0, P04p + Z0m + X4p --> Z0m + P04p + Z0m + X4p
    1.0, P04m + Z0p + X4p --> Z0m + P04m + Z0p + X4p
    1.0, P10p + Z1p + X0p --> Z1p + P10p + Z1p + X0p
    1.0, P10m + Z1m + X0p --> Z1p + P10m + Z1m + X0p
    1.0, P10p + Z1m + X0m --> Z1p + P10p + Z1m + X0m
    1.0, P10m + Z1p + X0m --> Z1p + P10m + Z1p + X0m
    1.0, P10p + Z1p + X0m --> Z1m + P10p + Z1p + X0m
    1.0, P10m + Z1m + X0m --> Z1m + P10m + Z1m + X0m
    1.0, P10p + Z1m + X0p --> Z1m + P10p + Z1m + X0p
    1.0, P10m + Z1p + X0p --> Z1m + P10m + Z1p + X0p
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
    1.0, P20p + Z2p + X0p --> Z2p + P20p + Z2p + X0p
    1.0, P20m + Z2m + X0p --> Z2p + P20m + Z2m + X0p
    1.0, P20p + Z2m + X0m --> Z2p + P20p + Z2m + X0m
    1.0, P20m + Z2p + X0m --> Z2p + P20m + Z2p + X0m
    1.0, P20p + Z2p + X0m --> Z2m + P20p + Z2p + X0m
    1.0, P20m + Z2m + X0m --> Z2m + P20m + Z2m + X0m
    1.0, P20p + Z2m + X0p --> Z2m + P20p + Z2m + X0p
    1.0, P20m + Z2p + X0p --> Z2m + P20m + Z2p + X0p
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
    1.0, P30p + Z3p + X0p --> Z3p + P30p + Z3p + X0p
    1.0, P30m + Z3m + X0p --> Z3p + P30m + Z3m + X0p
    1.0, P30p + Z3m + X0m --> Z3p + P30p + Z3m + X0m
    1.0, P30m + Z3p + X0m --> Z3p + P30m + Z3p + X0m
    1.0, P30p + Z3p + X0m --> Z3m + P30p + Z3p + X0m
    1.0, P30m + Z3m + X0m --> Z3m + P30m + Z3m + X0m
    1.0, P30p + Z3m + X0p --> Z3m + P30p + Z3m + X0p
    1.0, P30m + Z3p + X0p --> Z3m + P30m + Z3p + X0p
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
    1.0, P40p + Z4p + X0p --> Z4p + P40p + Z4p + X0p
    1.0, P40m + Z4m + X0p --> Z4p + P40m + Z4m + X0p
    1.0, P40p + Z4m + X0m --> Z4p + P40p + Z4m + X0m
    1.0, P40m + Z4p + X0m --> Z4p + P40m + Z4p + X0m
    1.0, P40p + Z4p + X0m --> Z4m + P40p + Z4p + X0m
    1.0, P40m + Z4m + X0m --> Z4m + P40m + Z4m + X0m
    1.0, P40p + Z4m + X0p --> Z4m + P40p + Z4m + X0p
    1.0, P40m + Z4p + X0p --> Z4m + P40m + Z4p + X0p
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
    1.0, B0p + Z0p --> Z0p + B0p + Z0p
    1.0, B0m + Z0m --> Z0p + B0m + Z0m
    1.0, B0p + Z0m --> Z0m + B0p + Z0m
    1.0, B0m + Z0p --> Z0m + B0m + Z0p
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
    100.0, Z0p + Z0m --> 0
    1.0, Z0p + Z0p --> Z0p
    1.0, Z0m + Z0m --> Z0m
    100.0, Z1p + Z1m --> 0
    1.0, Z1p + Z1p --> Z1p
    1.0, Z1m + Z1m --> Z1m
    100.0, Z2p + Z2m --> 0
    1.0, Z2p + Z2p --> Z2p
    1.0, Z2m + Z2m --> Z2m
    100.0, Z3p + Z3m --> 0
    1.0, Z3p + Z3p --> Z3p
    1.0, Z3m + Z3m --> Z3m
    100.0, Z4p + Z4m --> 0
    1.0, Z4p + Z4p --> Z4p
    1.0, Z4m + Z4m --> Z4m
end
rn_yhat_calculate = @reaction_network rn_yhat_calculate begin
    1.0, W00p + Z0p --> O0p + W00p + Z0p
    1.0, W00m + Z0m --> O0p + W00m + Z0m
    1.0, W00p + Z0m --> O0m + W00p + Z0m
    1.0, W00m + Z0p --> O0m + W00m + Z0p
    1.0, W10p + Z1p --> O0p + W10p + Z1p
    1.0, W10m + Z1m --> O0p + W10m + Z1m
    1.0, W10p + Z1m --> O0m + W10p + Z1m
    1.0, W10m + Z1p --> O0m + W10m + Z1p
    1.0, W20p + Z2p --> O0p + W20p + Z2p
    1.0, W20m + Z2m --> O0p + W20m + Z2m
    1.0, W20p + Z2m --> O0m + W20p + Z2m
    1.0, W20m + Z2p --> O0m + W20m + Z2p
    1.0, W30p + Z3p --> O0p + W30p + Z3p
    1.0, W30m + Z3m --> O0p + W30m + Z3m
    1.0, W30p + Z3m --> O0m + W30p + Z3m
    1.0, W30m + Z3p --> O0m + W30m + Z3p
    1.0, W40p + Z4p --> O0p + W40p + Z4p
    1.0, W40m + Z4m --> O0p + W40m + Z4m
    1.0, W40p + Z4m --> O0m + W40p + Z4m
    1.0, W40m + Z4p --> O0m + W40m + Z4p
    1.0, W01p + Z0p --> O1p + W01p + Z0p
    1.0, W01m + Z0m --> O1p + W01m + Z0m
    1.0, W01p + Z0m --> O1m + W01p + Z0m
    1.0, W01m + Z0p --> O1m + W01m + Z0p
    1.0, W11p + Z1p --> O1p + W11p + Z1p
    1.0, W11m + Z1m --> O1p + W11m + Z1m
    1.0, W11p + Z1m --> O1m + W11p + Z1m
    1.0, W11m + Z1p --> O1m + W11m + Z1p
    1.0, W21p + Z2p --> O1p + W21p + Z2p
    1.0, W21m + Z2m --> O1p + W21m + Z2m
    1.0, W21p + Z2m --> O1m + W21p + Z2m
    1.0, W21m + Z2p --> O1m + W21m + Z2p
    1.0, W31p + Z3p --> O1p + W31p + Z3p
    1.0, W31m + Z3m --> O1p + W31m + Z3m
    1.0, W31p + Z3m --> O1m + W31p + Z3m
    1.0, W31m + Z3p --> O1m + W31m + Z3p
    1.0, W41p + Z4p --> O1p + W41p + Z4p
    1.0, W41m + Z4m --> O1p + W41m + Z4m
    1.0, W41p + Z4m --> O1m + W41p + Z4m
    1.0, W41m + Z4p --> O1m + W41m + Z4p
    1.0, W02p + Z0p --> O2p + W02p + Z0p
    1.0, W02m + Z0m --> O2p + W02m + Z0m
    1.0, W02p + Z0m --> O2m + W02p + Z0m
    1.0, W02m + Z0p --> O2m + W02m + Z0p
    1.0, W12p + Z1p --> O2p + W12p + Z1p
    1.0, W12m + Z1m --> O2p + W12m + Z1m
    1.0, W12p + Z1m --> O2m + W12p + Z1m
    1.0, W12m + Z1p --> O2m + W12m + Z1p
    1.0, W22p + Z2p --> O2p + W22p + Z2p
    1.0, W22m + Z2m --> O2p + W22m + Z2m
    1.0, W22p + Z2m --> O2m + W22p + Z2m
    1.0, W22m + Z2p --> O2m + W22m + Z2p
    1.0, W32p + Z3p --> O2p + W32p + Z3p
    1.0, W32m + Z3m --> O2p + W32m + Z3m
    1.0, W32p + Z3m --> O2m + W32p + Z3m
    1.0, W32m + Z3p --> O2m + W32m + Z3p
    1.0, W42p + Z4p --> O2p + W42p + Z4p
    1.0, W42m + Z4m --> O2p + W42m + Z4m
    1.0, W42p + Z4m --> O2m + W42p + Z4m
    1.0, W42m + Z4p --> O2m + W42m + Z4p
    1.0, O0p --> 0
    1.0, O0m --> 0
    1.0, O1p --> 0
    1.0, O1m --> 0
    1.0, O2p --> 0
    1.0, O2m --> 0
end
# Calculate the adjoint
rn_adjoint_calculate = @reaction_network rn_adjoint_calculate begin
    1.0, W00p + E0p --> A0p + W00p + E0p
    1.0, W00m + E0m --> A0p + W00m + E0m
    1.0, W00p + E0m --> A0m + W00p + E0m
    1.0, W00m + E0p --> A0m + W00m + E0p
    1.0, W01p + E1p --> A0p + W01p + E1p
    1.0, W01m + E1m --> A0p + W01m + E1m
    1.0, W01p + E1m --> A0m + W01p + E1m
    1.0, W01m + E1p --> A0m + W01m + E1p
    1.0, W02p + E2p --> A0p + W02p + E2p
    1.0, W02m + E2m --> A0p + W02m + E2m
    1.0, W02p + E2m --> A0m + W02p + E2m
    1.0, W02m + E2p --> A0m + W02m + E2p
    1.0, W10p + E0p --> A1p + W10p + E0p
    1.0, W10m + E0m --> A1p + W10m + E0m
    1.0, W10p + E0m --> A1m + W10p + E0m
    1.0, W10m + E0p --> A1m + W10m + E0p
    1.0, W11p + E1p --> A1p + W11p + E1p
    1.0, W11m + E1m --> A1p + W11m + E1m
    1.0, W11p + E1m --> A1m + W11p + E1m
    1.0, W11m + E1p --> A1m + W11m + E1p
    1.0, W12p + E2p --> A1p + W12p + E2p
    1.0, W12m + E2m --> A1p + W12m + E2m
    1.0, W12p + E2m --> A1m + W12p + E2m
    1.0, W12m + E2p --> A1m + W12m + E2p
    1.0, W20p + E0p --> A2p + W20p + E0p
    1.0, W20m + E0m --> A2p + W20m + E0m
    1.0, W20p + E0m --> A2m + W20p + E0m
    1.0, W20m + E0p --> A2m + W20m + E0p
    1.0, W21p + E1p --> A2p + W21p + E1p
    1.0, W21m + E1m --> A2p + W21m + E1m
    1.0, W21p + E1m --> A2m + W21p + E1m
    1.0, W21m + E1p --> A2m + W21m + E1p
    1.0, W22p + E2p --> A2p + W22p + E2p
    1.0, W22m + E2m --> A2p + W22m + E2m
    1.0, W22p + E2m --> A2m + W22p + E2m
    1.0, W22m + E2p --> A2m + W22m + E2p
    1.0, W30p + E0p --> A3p + W30p + E0p
    1.0, W30m + E0m --> A3p + W30m + E0m
    1.0, W30p + E0m --> A3m + W30p + E0m
    1.0, W30m + E0p --> A3m + W30m + E0p
    1.0, W31p + E1p --> A3p + W31p + E1p
    1.0, W31m + E1m --> A3p + W31m + E1m
    1.0, W31p + E1m --> A3m + W31p + E1m
    1.0, W31m + E1p --> A3m + W31m + E1p
    1.0, W32p + E2p --> A3p + W32p + E2p
    1.0, W32m + E2m --> A3p + W32m + E2m
    1.0, W32p + E2m --> A3m + W32p + E2m
    1.0, W32m + E2p --> A3m + W32m + E2p
    1.0, W40p + E0p --> A4p + W40p + E0p
    1.0, W40m + E0m --> A4p + W40m + E0m
    1.0, W40p + E0m --> A4m + W40p + E0m
    1.0, W40m + E0p --> A4m + W40m + E0p
    1.0, W41p + E1p --> A4p + W41p + E1p
    1.0, W41m + E1m --> A4p + W41m + E1m
    1.0, W41p + E1m --> A4m + W41p + E1m
    1.0, W41m + E1p --> A4m + W41m + E1p
    1.0, W42p + E2p --> A4p + W42p + E2p
    1.0, W42m + E2m --> A4p + W42m + E2m
    1.0, W42p + E2m --> A4m + W42p + E2m
    1.0, W42m + E2p --> A4m + W42m + E2p
    1.0, A0p --> 0
    1.0, A0m --> 0
    1.0, A1p --> 0
    1.0, A1m --> 0
    1.0, A2p --> 0
    1.0, A2m --> 0
    1.0, A3p --> 0
    1.0, A3m --> 0
    1.0, A4p --> 0
    1.0, A4m --> 0
end
rn_dual_node_relu_bwd = @reaction_network rn_dual_node_relu_bwd begin
    # dz/dt = -h
    1.0, H0m --> Z0p + H0m
    1.0, H0p --> Z0m + H0p
    1.0, H1m --> Z1p + H1m
    1.0, H1p --> Z1m + H1p
    1.0, H2m --> Z2p + H2m
    1.0, H2p --> Z2m + H2p
    1.0, H3m --> Z3p + H3m
    1.0, H3p --> Z3m + H3p
    1.0, H4m --> Z4p + H4m
    1.0, H4p --> Z4m + H4p
    # dz/dt = -p_ij x_j z_i
    1.0, P00p + Z0p + X0m --> Z0p + P00p + Z0p + X0m
    1.0, P00m + Z0m + X0m --> Z0p + P00m + Z0m + X0m
    1.0, P00p + Z0m + X0p --> Z0p + P00p + Z0m + X0p
    1.0, P00m + Z0p + X0p --> Z0p + P00m + Z0p + X0p
    1.0, P00p + Z0p + X0p --> Z0m + P00p + Z0p + X0p
    1.0, P00m + Z0m + X0p --> Z0m + P00m + Z0m + X0p
    1.0, P00p + Z0m + X0m --> Z0m + P00p + Z0m + X0m
    1.0, P00m + Z0p + X0m --> Z0m + P00m + Z0p + X0m
    1.0, P01p + Z0p + X1m --> Z0p + P01p + Z0p + X1m
    1.0, P01m + Z0m + X1m --> Z0p + P01m + Z0m + X1m
    1.0, P01p + Z0m + X1p --> Z0p + P01p + Z0m + X1p
    1.0, P01m + Z0p + X1p --> Z0p + P01m + Z0p + X1p
    1.0, P01p + Z0p + X1p --> Z0m + P01p + Z0p + X1p
    1.0, P01m + Z0m + X1p --> Z0m + P01m + Z0m + X1p
    1.0, P01p + Z0m + X1m --> Z0m + P01p + Z0m + X1m
    1.0, P01m + Z0p + X1m --> Z0m + P01m + Z0p + X1m
    1.0, P02p + Z0p + X2m --> Z0p + P02p + Z0p + X2m
    1.0, P02m + Z0m + X2m --> Z0p + P02m + Z0m + X2m
    1.0, P02p + Z0m + X2p --> Z0p + P02p + Z0m + X2p
    1.0, P02m + Z0p + X2p --> Z0p + P02m + Z0p + X2p
    1.0, P02p + Z0p + X2p --> Z0m + P02p + Z0p + X2p
    1.0, P02m + Z0m + X2p --> Z0m + P02m + Z0m + X2p
    1.0, P02p + Z0m + X2m --> Z0m + P02p + Z0m + X2m
    1.0, P02m + Z0p + X2m --> Z0m + P02m + Z0p + X2m
    1.0, P03p + Z0p + X3m --> Z0p + P03p + Z0p + X3m
    1.0, P03m + Z0m + X3m --> Z0p + P03m + Z0m + X3m
    1.0, P03p + Z0m + X3p --> Z0p + P03p + Z0m + X3p
    1.0, P03m + Z0p + X3p --> Z0p + P03m + Z0p + X3p
    1.0, P03p + Z0p + X3p --> Z0m + P03p + Z0p + X3p
    1.0, P03m + Z0m + X3p --> Z0m + P03m + Z0m + X3p
    1.0, P03p + Z0m + X3m --> Z0m + P03p + Z0m + X3m
    1.0, P03m + Z0p + X3m --> Z0m + P03m + Z0p + X3m
    1.0, P04p + Z0p + X4m --> Z0p + P04p + Z0p + X4m
    1.0, P04m + Z0m + X4m --> Z0p + P04m + Z0m + X4m
    1.0, P04p + Z0m + X4p --> Z0p + P04p + Z0m + X4p
    1.0, P04m + Z0p + X4p --> Z0p + P04m + Z0p + X4p
    1.0, P04p + Z0p + X4p --> Z0m + P04p + Z0p + X4p
    1.0, P04m + Z0m + X4p --> Z0m + P04m + Z0m + X4p
    1.0, P04p + Z0m + X4m --> Z0m + P04p + Z0m + X4m
    1.0, P04m + Z0p + X4m --> Z0m + P04m + Z0p + X4m
    1.0, P10p + Z1p + X0m --> Z1p + P10p + Z1p + X0m
    1.0, P10m + Z1m + X0m --> Z1p + P10m + Z1m + X0m
    1.0, P10p + Z1m + X0p --> Z1p + P10p + Z1m + X0p
    1.0, P10m + Z1p + X0p --> Z1p + P10m + Z1p + X0p
    1.0, P10p + Z1p + X0p --> Z1m + P10p + Z1p + X0p
    1.0, P10m + Z1m + X0p --> Z1m + P10m + Z1m + X0p
    1.0, P10p + Z1m + X0m --> Z1m + P10p + Z1m + X0m
    1.0, P10m + Z1p + X0m --> Z1m + P10m + Z1p + X0m
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
    1.0, P20p + Z2p + X0m --> Z2p + P20p + Z2p + X0m
    1.0, P20m + Z2m + X0m --> Z2p + P20m + Z2m + X0m
    1.0, P20p + Z2m + X0p --> Z2p + P20p + Z2m + X0p
    1.0, P20m + Z2p + X0p --> Z2p + P20m + Z2p + X0p
    1.0, P20p + Z2p + X0p --> Z2m + P20p + Z2p + X0p
    1.0, P20m + Z2m + X0p --> Z2m + P20m + Z2m + X0p
    1.0, P20p + Z2m + X0m --> Z2m + P20p + Z2m + X0m
    1.0, P20m + Z2p + X0m --> Z2m + P20m + Z2p + X0m
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
    1.0, P30p + Z3p + X0m --> Z3p + P30p + Z3p + X0m
    1.0, P30m + Z3m + X0m --> Z3p + P30m + Z3m + X0m
    1.0, P30p + Z3m + X0p --> Z3p + P30p + Z3m + X0p
    1.0, P30m + Z3p + X0p --> Z3p + P30m + Z3p + X0p
    1.0, P30p + Z3p + X0p --> Z3m + P30p + Z3p + X0p
    1.0, P30m + Z3m + X0p --> Z3m + P30m + Z3m + X0p
    1.0, P30p + Z3m + X0m --> Z3m + P30p + Z3m + X0m
    1.0, P30m + Z3p + X0m --> Z3m + P30m + Z3p + X0m
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
    1.0, P40p + Z4p + X0m --> Z4p + P40p + Z4p + X0m
    1.0, P40m + Z4m + X0m --> Z4p + P40m + Z4m + X0m
    1.0, P40p + Z4m + X0p --> Z4p + P40p + Z4m + X0p
    1.0, P40m + Z4p + X0p --> Z4p + P40m + Z4p + X0p
    1.0, P40p + Z4p + X0p --> Z4m + P40p + Z4p + X0p
    1.0, P40m + Z4m + X0p --> Z4m + P40m + Z4m + X0p
    1.0, P40p + Z4m + X0m --> Z4m + P40p + Z4m + X0m
    1.0, P40m + Z4p + X0m --> Z4m + P40m + Z4p + X0m
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
    1.0, B0p + Z0m --> Z0p + B0p + Z0m
    1.0, B0m + Z0p --> Z0p + B0m + Z0p
    1.0, B0p + Z0p --> Z0m + B0p + Z0p
    1.0, B0m + Z0m --> Z0m + B0m + Z0m
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
    100.0, Z0p + Z0m --> 0
    1.0, Z0p + Z0p --> Z0p + Z0p + Z0p
    1.0, Z0m + Z0m --> Z0m + Z0m + Z0m
    100.0, Z1p + Z1m --> 0
    1.0, Z1p + Z1p --> Z1p + Z1p + Z1p
    1.0, Z1m + Z1m --> Z1m + Z1m + Z1m
    100.0, Z2p + Z2m --> 0
    1.0, Z2p + Z2p --> Z2p + Z2p + Z2p
    1.0, Z2m + Z2m --> Z2m + Z2m + Z2m
    100.0, Z3p + Z3m --> 0
    1.0, Z3p + Z3p --> Z3p + Z3p + Z3p
    1.0, Z3m + Z3m --> Z3m + Z3m + Z3m
    100.0, Z4p + Z4m --> 0
    1.0, Z4p + Z4p --> Z4p + Z4p + Z4p
    1.0, Z4m + Z4m --> Z4m + Z4m + Z4m

    # da_i/dt = a_i p_ij x_j
    1.0, A0p + P00p + X0p --> A0p + A0p + P00p + X0p
    1.0, A0m + P00m + X0p --> A0p + A0m + P00m + X0p
    1.0, A0p + P00m + X0m --> A0p + A0p + P00m + X0m
    1.0, A0m + P00p + X0m --> A0p + A0m + P00p + X0m
    1.0, A0p + P00p + X0m --> A0m + A0p + P00p + X0m
    1.0, A0m + P00m + X0m --> A0m + A0m + P00m + X0m
    1.0, A0p + P00m + X0p --> A0m + A0p + P00m + X0p
    1.0, A0m + P00p + X0p --> A0m + A0m + P00p + X0p
    1.0, A0p + P01p + X1p --> A0p + A0p + P01p + X1p
    1.0, A0m + P01m + X1p --> A0p + A0m + P01m + X1p
    1.0, A0p + P01m + X1m --> A0p + A0p + P01m + X1m
    1.0, A0m + P01p + X1m --> A0p + A0m + P01p + X1m
    1.0, A0p + P01p + X1m --> A0m + A0p + P01p + X1m
    1.0, A0m + P01m + X1m --> A0m + A0m + P01m + X1m
    1.0, A0p + P01m + X1p --> A0m + A0p + P01m + X1p
    1.0, A0m + P01p + X1p --> A0m + A0m + P01p + X1p
    1.0, A0p + P02p + X2p --> A0p + A0p + P02p + X2p
    1.0, A0m + P02m + X2p --> A0p + A0m + P02m + X2p
    1.0, A0p + P02m + X2m --> A0p + A0p + P02m + X2m
    1.0, A0m + P02p + X2m --> A0p + A0m + P02p + X2m
    1.0, A0p + P02p + X2m --> A0m + A0p + P02p + X2m
    1.0, A0m + P02m + X2m --> A0m + A0m + P02m + X2m
    1.0, A0p + P02m + X2p --> A0m + A0p + P02m + X2p
    1.0, A0m + P02p + X2p --> A0m + A0m + P02p + X2p
    1.0, A0p + P03p + X3p --> A0p + A0p + P03p + X3p
    1.0, A0m + P03m + X3p --> A0p + A0m + P03m + X3p
    1.0, A0p + P03m + X3m --> A0p + A0p + P03m + X3m
    1.0, A0m + P03p + X3m --> A0p + A0m + P03p + X3m
    1.0, A0p + P03p + X3m --> A0m + A0p + P03p + X3m
    1.0, A0m + P03m + X3m --> A0m + A0m + P03m + X3m
    1.0, A0p + P03m + X3p --> A0m + A0p + P03m + X3p
    1.0, A0m + P03p + X3p --> A0m + A0m + P03p + X3p
    1.0, A0p + P04p + X4p --> A0p + A0p + P04p + X4p
    1.0, A0m + P04m + X4p --> A0p + A0m + P04m + X4p
    1.0, A0p + P04m + X4m --> A0p + A0p + P04m + X4m
    1.0, A0m + P04p + X4m --> A0p + A0m + P04p + X4m
    1.0, A0p + P04p + X4m --> A0m + A0p + P04p + X4m
    1.0, A0m + P04m + X4m --> A0m + A0m + P04m + X4m
    1.0, A0p + P04m + X4p --> A0m + A0p + P04m + X4p
    1.0, A0m + P04p + X4p --> A0m + A0m + P04p + X4p
    1.0, A1p + P10p + X0p --> A1p + A1p + P10p + X0p
    1.0, A1m + P10m + X0p --> A1p + A1m + P10m + X0p
    1.0, A1p + P10m + X0m --> A1p + A1p + P10m + X0m
    1.0, A1m + P10p + X0m --> A1p + A1m + P10p + X0m
    1.0, A1p + P10p + X0m --> A1m + A1p + P10p + X0m
    1.0, A1m + P10m + X0m --> A1m + A1m + P10m + X0m
    1.0, A1p + P10m + X0p --> A1m + A1p + P10m + X0p
    1.0, A1m + P10p + X0p --> A1m + A1m + P10p + X0p
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
    1.0, A2p + P20p + X0p --> A2p + A2p + P20p + X0p
    1.0, A2m + P20m + X0p --> A2p + A2m + P20m + X0p
    1.0, A2p + P20m + X0m --> A2p + A2p + P20m + X0m
    1.0, A2m + P20p + X0m --> A2p + A2m + P20p + X0m
    1.0, A2p + P20p + X0m --> A2m + A2p + P20p + X0m
    1.0, A2m + P20m + X0m --> A2m + A2m + P20m + X0m
    1.0, A2p + P20m + X0p --> A2m + A2p + P20m + X0p
    1.0, A2m + P20p + X0p --> A2m + A2m + P20p + X0p
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
    1.0, A3p + P30p + X0p --> A3p + A3p + P30p + X0p
    1.0, A3m + P30m + X0p --> A3p + A3m + P30m + X0p
    1.0, A3p + P30m + X0m --> A3p + A3p + P30m + X0m
    1.0, A3m + P30p + X0m --> A3p + A3m + P30p + X0m
    1.0, A3p + P30p + X0m --> A3m + A3p + P30p + X0m
    1.0, A3m + P30m + X0m --> A3m + A3m + P30m + X0m
    1.0, A3p + P30m + X0p --> A3m + A3p + P30m + X0p
    1.0, A3m + P30p + X0p --> A3m + A3m + P30p + X0p
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
    1.0, A4p + P40p + X0p --> A4p + A4p + P40p + X0p
    1.0, A4m + P40m + X0p --> A4p + A4m + P40m + X0p
    1.0, A4p + P40m + X0m --> A4p + A4p + P40m + X0m
    1.0, A4m + P40p + X0m --> A4p + A4m + P40p + X0m
    1.0, A4p + P40p + X0m --> A4m + A4p + P40p + X0m
    1.0, A4m + P40m + X0m --> A4m + A4m + P40m + X0m
    1.0, A4p + P40m + X0p --> A4m + A4p + P40m + X0p
    1.0, A4m + P40p + X0p --> A4m + A4m + P40p + X0p
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
    1.0, A0p + Z0p + X0p --> G00p + A0p + Z0p + X0p
    1.0, A0m + Z0m + X0p --> G00p + A0m + Z0m + X0p
    1.0, A0p + Z0m + X0m --> G00p + A0p + Z0m + X0m
    1.0, A0m + Z0p + X0m --> G00p + A0m + Z0p + X0m
    1.0, A0p + Z0p + X0m --> G00m + A0p + Z0p + X0m
    1.0, A0m + Z0m + X0m --> G00m + A0m + Z0m + X0m
    1.0, A0p + Z0m + X0p --> G00m + A0p + Z0m + X0p
    1.0, A0m + Z0p + X0p --> G00m + A0m + Z0p + X0p
    1.0, A0p + Z0p + X1p --> G01p + A0p + Z0p + X1p
    1.0, A0m + Z0m + X1p --> G01p + A0m + Z0m + X1p
    1.0, A0p + Z0m + X1m --> G01p + A0p + Z0m + X1m
    1.0, A0m + Z0p + X1m --> G01p + A0m + Z0p + X1m
    1.0, A0p + Z0p + X1m --> G01m + A0p + Z0p + X1m
    1.0, A0m + Z0m + X1m --> G01m + A0m + Z0m + X1m
    1.0, A0p + Z0m + X1p --> G01m + A0p + Z0m + X1p
    1.0, A0m + Z0p + X1p --> G01m + A0m + Z0p + X1p
    1.0, A0p + Z0p + X2p --> G02p + A0p + Z0p + X2p
    1.0, A0m + Z0m + X2p --> G02p + A0m + Z0m + X2p
    1.0, A0p + Z0m + X2m --> G02p + A0p + Z0m + X2m
    1.0, A0m + Z0p + X2m --> G02p + A0m + Z0p + X2m
    1.0, A0p + Z0p + X2m --> G02m + A0p + Z0p + X2m
    1.0, A0m + Z0m + X2m --> G02m + A0m + Z0m + X2m
    1.0, A0p + Z0m + X2p --> G02m + A0p + Z0m + X2p
    1.0, A0m + Z0p + X2p --> G02m + A0m + Z0p + X2p
    1.0, A0p + Z0p + X3p --> G03p + A0p + Z0p + X3p
    1.0, A0m + Z0m + X3p --> G03p + A0m + Z0m + X3p
    1.0, A0p + Z0m + X3m --> G03p + A0p + Z0m + X3m
    1.0, A0m + Z0p + X3m --> G03p + A0m + Z0p + X3m
    1.0, A0p + Z0p + X3m --> G03m + A0p + Z0p + X3m
    1.0, A0m + Z0m + X3m --> G03m + A0m + Z0m + X3m
    1.0, A0p + Z0m + X3p --> G03m + A0p + Z0m + X3p
    1.0, A0m + Z0p + X3p --> G03m + A0m + Z0p + X3p
    1.0, A0p + Z0p + X4p --> G04p + A0p + Z0p + X4p
    1.0, A0m + Z0m + X4p --> G04p + A0m + Z0m + X4p
    1.0, A0p + Z0m + X4m --> G04p + A0p + Z0m + X4m
    1.0, A0m + Z0p + X4m --> G04p + A0m + Z0p + X4m
    1.0, A0p + Z0p + X4m --> G04m + A0p + Z0p + X4m
    1.0, A0m + Z0m + X4m --> G04m + A0m + Z0m + X4m
    1.0, A0p + Z0m + X4p --> G04m + A0p + Z0m + X4p
    1.0, A0m + Z0p + X4p --> G04m + A0m + Z0p + X4p
    1.0, A1p + Z1p + X0p --> G10p + A1p + Z1p + X0p
    1.0, A1m + Z1m + X0p --> G10p + A1m + Z1m + X0p
    1.0, A1p + Z1m + X0m --> G10p + A1p + Z1m + X0m
    1.0, A1m + Z1p + X0m --> G10p + A1m + Z1p + X0m
    1.0, A1p + Z1p + X0m --> G10m + A1p + Z1p + X0m
    1.0, A1m + Z1m + X0m --> G10m + A1m + Z1m + X0m
    1.0, A1p + Z1m + X0p --> G10m + A1p + Z1m + X0p
    1.0, A1m + Z1p + X0p --> G10m + A1m + Z1p + X0p
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
    1.0, A2p + Z2p + X0p --> G20p + A2p + Z2p + X0p
    1.0, A2m + Z2m + X0p --> G20p + A2m + Z2m + X0p
    1.0, A2p + Z2m + X0m --> G20p + A2p + Z2m + X0m
    1.0, A2m + Z2p + X0m --> G20p + A2m + Z2p + X0m
    1.0, A2p + Z2p + X0m --> G20m + A2p + Z2p + X0m
    1.0, A2m + Z2m + X0m --> G20m + A2m + Z2m + X0m
    1.0, A2p + Z2m + X0p --> G20m + A2p + Z2m + X0p
    1.0, A2m + Z2p + X0p --> G20m + A2m + Z2p + X0p
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
    1.0, A3p + Z3p + X0p --> G30p + A3p + Z3p + X0p
    1.0, A3m + Z3m + X0p --> G30p + A3m + Z3m + X0p
    1.0, A3p + Z3m + X0m --> G30p + A3p + Z3m + X0m
    1.0, A3m + Z3p + X0m --> G30p + A3m + Z3p + X0m
    1.0, A3p + Z3p + X0m --> G30m + A3p + Z3p + X0m
    1.0, A3m + Z3m + X0m --> G30m + A3m + Z3m + X0m
    1.0, A3p + Z3m + X0p --> G30m + A3p + Z3m + X0p
    1.0, A3m + Z3p + X0p --> G30m + A3m + Z3p + X0p
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
    1.0, A4p + Z4p + X0p --> G40p + A4p + Z4p + X0p
    1.0, A4m + Z4m + X0p --> G40p + A4m + Z4m + X0p
    1.0, A4p + Z4m + X0m --> G40p + A4p + Z4m + X0m
    1.0, A4m + Z4p + X0m --> G40p + A4m + Z4p + X0m
    1.0, A4p + Z4p + X0m --> G40m + A4p + Z4p + X0m
    1.0, A4m + Z4m + X0m --> G40m + A4m + Z4m + X0m
    1.0, A4p + Z4m + X0p --> G40m + A4p + Z4m + X0p
    1.0, A4m + Z4p + X0p --> G40m + A4m + Z4p + X0p
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
    1.0, A0p + Z0p --> V0p + A0p + Z0p
    1.0, A0m + Z0m --> V0p + A0m + Z0m
    1.0, A0p + Z0m --> V0m + A0p + Z0m
    1.0, A0m + Z0p --> V0m + A0m + Z0p
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
# Calculate wgrads
rn_wgrads_calculate = @reaction_network rn_wgrads_calculate begin
    1.0, Z0p + E0p --> M00p + Z0p + E0p
    1.0, Z0m + E0m --> M00p + Z0m + E0m
    1.0, Z0p + E0m --> M00m + Z0p + E0m
    1.0, Z0m + E0p --> M00m + Z0m + E0p
    1.0, Z0p + E1p --> M01p + Z0p + E1p
    1.0, Z0m + E1m --> M01p + Z0m + E1m
    1.0, Z0p + E1m --> M01m + Z0p + E1m
    1.0, Z0m + E1p --> M01m + Z0m + E1p
    1.0, Z0p + E2p --> M02p + Z0p + E2p
    1.0, Z0m + E2m --> M02p + Z0m + E2m
    1.0, Z0p + E2m --> M02m + Z0p + E2m
    1.0, Z0m + E2p --> M02m + Z0m + E2p
    1.0, Z1p + E0p --> M10p + Z1p + E0p
    1.0, Z1m + E0m --> M10p + Z1m + E0m
    1.0, Z1p + E0m --> M10m + Z1p + E0m
    1.0, Z1m + E0p --> M10m + Z1m + E0p
    1.0, Z1p + E1p --> M11p + Z1p + E1p
    1.0, Z1m + E1m --> M11p + Z1m + E1m
    1.0, Z1p + E1m --> M11m + Z1p + E1m
    1.0, Z1m + E1p --> M11m + Z1m + E1p
    1.0, Z1p + E2p --> M12p + Z1p + E2p
    1.0, Z1m + E2m --> M12p + Z1m + E2m
    1.0, Z1p + E2m --> M12m + Z1p + E2m
    1.0, Z1m + E2p --> M12m + Z1m + E2p
    1.0, Z2p + E0p --> M20p + Z2p + E0p
    1.0, Z2m + E0m --> M20p + Z2m + E0m
    1.0, Z2p + E0m --> M20m + Z2p + E0m
    1.0, Z2m + E0p --> M20m + Z2m + E0p
    1.0, Z2p + E1p --> M21p + Z2p + E1p
    1.0, Z2m + E1m --> M21p + Z2m + E1m
    1.0, Z2p + E1m --> M21m + Z2p + E1m
    1.0, Z2m + E1p --> M21m + Z2m + E1p
    1.0, Z2p + E2p --> M22p + Z2p + E2p
    1.0, Z2m + E2m --> M22p + Z2m + E2m
    1.0, Z2p + E2m --> M22m + Z2p + E2m
    1.0, Z2m + E2p --> M22m + Z2m + E2p
    1.0, Z3p + E0p --> M30p + Z3p + E0p
    1.0, Z3m + E0m --> M30p + Z3m + E0m
    1.0, Z3p + E0m --> M30m + Z3p + E0m
    1.0, Z3m + E0p --> M30m + Z3m + E0p
    1.0, Z3p + E1p --> M31p + Z3p + E1p
    1.0, Z3m + E1m --> M31p + Z3m + E1m
    1.0, Z3p + E1m --> M31m + Z3p + E1m
    1.0, Z3m + E1p --> M31m + Z3m + E1p
    1.0, Z3p + E2p --> M32p + Z3p + E2p
    1.0, Z3m + E2m --> M32p + Z3m + E2m
    1.0, Z3p + E2m --> M32m + Z3p + E2m
    1.0, Z3m + E2p --> M32m + Z3m + E2p
    1.0, Z4p + E0p --> M40p + Z4p + E0p
    1.0, Z4m + E0m --> M40p + Z4m + E0m
    1.0, Z4p + E0m --> M40m + Z4p + E0m
    1.0, Z4m + E0p --> M40m + Z4m + E0p
    1.0, Z4p + E1p --> M41p + Z4p + E1p
    1.0, Z4m + E1m --> M41p + Z4m + E1m
    1.0, Z4p + E1m --> M41m + Z4p + E1m
    1.0, Z4m + E1p --> M41m + Z4m + E1p
    1.0, Z4p + E2p --> M42p + Z4p + E2p
    1.0, Z4m + E2m --> M42p + Z4m + E2m
    1.0, Z4p + E2m --> M42m + Z4p + E2m
    1.0, Z4m + E2p --> M42m + Z4m + E2p
    1.0, M00p --> 0
    1.0, M00m --> 0
    1.0, M01p --> 0
    1.0, M01m --> 0
    1.0, M02p --> 0
    1.0, M02m --> 0
    1.0, M10p --> 0
    1.0, M10m --> 0
    1.0, M11p --> 0
    1.0, M11m --> 0
    1.0, M12p --> 0
    1.0, M12m --> 0
    1.0, M20p --> 0
    1.0, M20m --> 0
    1.0, M21p --> 0
    1.0, M21m --> 0
    1.0, M22p --> 0
    1.0, M22m --> 0
    1.0, M30p --> 0
    1.0, M30m --> 0
    1.0, M31p --> 0
    1.0, M31m --> 0
    1.0, M32p --> 0
    1.0, M32m --> 0
    1.0, M40p --> 0
    1.0, M40m --> 0
    1.0, M41p --> 0
    1.0, M41m --> 0
    1.0, M42p --> 0
    1.0, M42m --> 0
end
# dy/dt = ab. Add Yp --> 0 and Ym --> 0
rn_dual_dot = @reaction_network rn_dual_dot begin
    1.0, A0p + B0p --> Y0p + A0p + B0p
    1.0, A0m + B0m --> Y0p + A0m + B0m
    1.0, A0p + B0m --> Y0m + A0p + B0m
    1.0, A0m + B0p --> Y0m + A0m + B0p
    1.0, A1p + B1p --> Y1p + A1p + B1p
    1.0, A1m + B1m --> Y1p + A1m + B1m
    1.0, A1p + B1m --> Y1m + A1p + B1m
    1.0, A1m + B1p --> Y1m + A1m + B1p
    1.0, A2p + B2p --> Y2p + A2p + B2p
    1.0, A2m + B2m --> Y2p + A2m + B2m
    1.0, A2p + B2m --> Y2m + A2p + B2m
    1.0, A2m + B2p --> Y2m + A2m + B2p
    1.0, A3p + B3p --> Y3p + A3p + B3p
    1.0, A3m + B3m --> Y3p + A3m + B3m
    1.0, A3p + B3m --> Y3m + A3p + B3m
    1.0, A3m + B3p --> Y3m + A3m + B3p
    1.0, A4p + B4p --> Y4p + A4p + B4p
    1.0, A4m + B4m --> Y4p + A4m + B4m
    1.0, A4p + B4m --> Y4m + A4p + B4m
    1.0, A4m + B4p --> Y4m + A4m + B4p
end
# rn_param_update
rn_param_update = @reaction_network rn_param_update begin
    k1, G00p --> P00m
    k1, G00m --> P00p
    k2, G00p --> 0
    k2, G00m --> 0
    k1, G01p --> P01m
    k1, G01m --> P01p
    k2, G01p --> 0
    k2, G01m --> 0
    k1, G02p --> P02m
    k1, G02m --> P02p
    k2, G02p --> 0
    k2, G02m --> 0
    k1, G03p --> P03m
    k1, G03m --> P03p
    k2, G03p --> 0
    k2, G03m --> 0
    k1, G04p --> P04m
    k1, G04m --> P04p
    k2, G04p --> 0
    k2, G04m --> 0
    k1, G10p --> P10m
    k1, G10m --> P10p
    k2, G10p --> 0
    k2, G10m --> 0
    k1, G11p --> P11m
    k1, G11m --> P11p
    k2, G11p --> 0
    k2, G11m --> 0
    k1, G12p --> P12m
    k1, G12m --> P12p
    k2, G12p --> 0
    k2, G12m --> 0
    k1, G13p --> P13m
    k1, G13m --> P13p
    k2, G13p --> 0
    k2, G13m --> 0
    k1, G14p --> P14m
    k1, G14m --> P14p
    k2, G14p --> 0
    k2, G14m --> 0
    k1, G20p --> P20m
    k1, G20m --> P20p
    k2, G20p --> 0
    k2, G20m --> 0
    k1, G21p --> P21m
    k1, G21m --> P21p
    k2, G21p --> 0
    k2, G21m --> 0
    k1, G22p --> P22m
    k1, G22m --> P22p
    k2, G22p --> 0
    k2, G22m --> 0
    k1, G23p --> P23m
    k1, G23m --> P23p
    k2, G23p --> 0
    k2, G23m --> 0
    k1, G24p --> P24m
    k1, G24m --> P24p
    k2, G24p --> 0
    k2, G24m --> 0
    k1, G30p --> P30m
    k1, G30m --> P30p
    k2, G30p --> 0
    k2, G30m --> 0
    k1, G31p --> P31m
    k1, G31m --> P31p
    k2, G31p --> 0
    k2, G31m --> 0
    k1, G32p --> P32m
    k1, G32m --> P32p
    k2, G32p --> 0
    k2, G32m --> 0
    k1, G33p --> P33m
    k1, G33m --> P33p
    k2, G33p --> 0
    k2, G33m --> 0
    k1, G34p --> P34m
    k1, G34m --> P34p
    k2, G34p --> 0
    k2, G34m --> 0
    k1, G40p --> P40m
    k1, G40m --> P40p
    k2, G40p --> 0
    k2, G40m --> 0
    k1, G41p --> P41m
    k1, G41m --> P41p
    k2, G41p --> 0
    k2, G41m --> 0
    k1, G42p --> P42m
    k1, G42m --> P42p
    k2, G42p --> 0
    k2, G42m --> 0
    k1, G43p --> P43m
    k1, G43m --> P43p
    k2, G43p --> 0
    k2, G43m --> 0
    k1, G44p --> P44m
    k1, G44m --> P44p
    k2, G44p --> 0
    k2, G44m --> 0
    k1, V0p --> B0m
    k1, V0m --> B0p
    k2, V0p --> 0
    k2, V0m --> 0
    k1, V1p --> B1m
    k1, V1m --> B1p
    k2, V1p --> 0
    k2, V1m --> 0
    k1, V2p --> B2m
    k1, V2m --> B2p
    k2, V2p --> 0
    k2, V2m --> 0
    k1, V3p --> B3m
    k1, V3m --> B3p
    k2, V3p --> 0
    k2, V3m --> 0
    k1, V4p --> B4m
    k1, V4m --> B4p
    k2, V4p --> 0
    k2, V4m --> 0
end
# rn_final_layer_update
rn_final_layer_update = @reaction_network rn_final_layer_update begin
    k1, M00p --> W00m
    k1, M00m --> W00p
    k2, M00p --> 0
    k2, M00m --> 0
    k1, M01p --> W01m
    k1, M01m --> W01p
    k2, M01p --> 0
    k2, M01m --> 0
    k1, M02p --> W02m
    k1, M02m --> W02p
    k2, M02p --> 0
    k2, M02m --> 0
    k1, M10p --> W10m
    k1, M10m --> W10p
    k2, M10p --> 0
    k2, M10m --> 0
    k1, M11p --> W11m
    k1, M11m --> W11p
    k2, M11p --> 0
    k2, M11m --> 0
    k1, M12p --> W12m
    k1, M12m --> W12p
    k2, M12p --> 0
    k2, M12m --> 0
    k1, M20p --> W20m
    k1, M20m --> W20p
    k2, M20p --> 0
    k2, M20m --> 0
    k1, M21p --> W21m
    k1, M21m --> W21p
    k2, M21p --> 0
    k2, M21m --> 0
    k1, M22p --> W22m
    k1, M22m --> W22p
    k2, M22p --> 0
    k2, M22m --> 0
    k1, M30p --> W30m
    k1, M30m --> W30p
    k2, M30p --> 0
    k2, M30m --> 0
    k1, M31p --> W31m
    k1, M31m --> W31p
    k2, M31p --> 0
    k2, M31m --> 0
    k1, M32p --> W32m
    k1, M32m --> W32p
    k2, M32p --> 0
    k2, M32m --> 0
    k1, M40p --> W40m
    k1, M40m --> W40p
    k2, M40p --> 0
    k2, M40m --> 0
    k1, M41p --> W41m
    k1, M41m --> W41p
    k2, M41p --> 0
    k2, M41m --> 0
    k1, M42p --> W42m
    k1, M42m --> W42p
    k2, M42p --> 0
    k2, M42m --> 0
end
# Create error species
rn_create_error_species = @reaction_network rn_create_error_species begin
    10.0, O0p --> E0p
    10.0, Y0p --> E0m
    10.0, O0m --> E0m
    10.0, Y0m --> E0p
    100.0, E0p + E0m --> 0
    10.0, O1p --> E1p
    10.0, Y1p --> E1m
    10.0, O1m --> E1m
    10.0, Y1m --> E1p
    100.0, E1p + E1m --> 0
    10.0, O2p --> E2p
    10.0, Y2p --> E2m
    10.0, O2m --> E2m
    10.0, Y2m --> E2p
    100.0, E2p + E2m --> 0
end
