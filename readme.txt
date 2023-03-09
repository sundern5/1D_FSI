READ ME

To run the program, adjust the material properties of the vessel and the flow profile (The built in unction will generate a gaussian distribution pulse).

The pressures are provided only to be used for BC calculations.

Changes 9/21/2022 (From main branch):

1. Main Script changed to FSI 1_D function that takes material stiffness, resistance, and flow profile 

2. Solver function moved to side_functions.py

3. Added param_sweep.py as a means to redo the siulation with different values of stiffness/resistance.

4. Added ability to record every n-th step to reduce writing cost.

5. Added param sweep plotter to plot the variation of peak pressure with change in parameters

Changes 3/9/2023 

1.  Saving changes before material retool

