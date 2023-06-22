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

3. Edited functions to make the code work for constant radius.

4. Simulation is stable and runs to completion.


Next steps:

1. Store P, B and derivatives as a function of area ratio, then interpolate during calls.





