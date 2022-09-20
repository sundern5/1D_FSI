READ ME

To run the program, adjust the material properties of the vessel and the flow profile (The built in unction will generate a gaussian distribution pulse).

The pressures are provided only to be used for BC calculations.

Changes from Main:

1. The solver logic has been changed in sn_dev1. The solver will go through every time-point multiple times to reduce error before moving to next time point.

2. Pressure equations and derivatives have been corrected based on literature.

3. Progress tracker has been added (Based on current simulation time point and end time point).