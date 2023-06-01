READ ME

This is the SN_DEV2 version. Planned changes:

1.  Implement non-linear material behavior
    a. Redefine tube class functions P, dPdA and dPdx1
    b. Redefine tube class functions B, Bh, dBdAh, dBdx1, dBdx1h, d2BdAdxh
    c. Redefine CFL condition (Not properly done yet)
    d. Redefine wave speed cnst (same as CFL)

CURRENT STATUS:

1. Non-linear behavior implemented.

2. Relevant functions updated (except CFD and cnst)

3. Reduced tolerances in Newton-Raphson method in bifurcation function (currently set to 1e-4)

3. Simulation is stable and runs to completion when fiber terms are not included.

4. Simulation runs up to 1.5% when fiber terms are included

5. Altered bifurcation tolerances.

Next steps:

1. Replace exponential function with taylor series expansion




