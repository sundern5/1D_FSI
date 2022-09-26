import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import side_functions as ic
from Tube_class import artery
from Main import FSI_1D

# Define vessel mechanical properties =======================# !! NEED TO MODIFY THIS !!

# Eh/r0 = k1*exp(k2*r0) + k3
# where E, h, and r0 are the Youngs Modulus, wall thickness, and reference radius, respectively.

k1 = 1e+5
k2 = -25
k3 = 93+4           #[1e+4, 9e+4, 2e+5, 4e+5, 6e+5, 8e+5, 9e+5]

kvals = [k1, k2, k3]       ## supposed elastic parameters for function

# Define flow waveform ======================================#

# read flow rate from csv file -> Comment out if not provided
# Qin = np.array(np.genfromtxt('flow_profile.csv', usecols=1))
# tQ = np.array(np.genfromtxt('flow_profile.csv', usecols=0))

#============================================================#

Qmin = 0        # Min flow rate (ml/s)
Qmax = 50      # Max flow rate (ml/s)

t0 = 0.0        # initial time (s)
t_end = 1       # Duration of 1 cardiac cycle (s)

t_systole = 0.15        # Peak systole time (s)
t_diastole = 0.4        # Peak diastole time (s)

Qin, tQ = ic.inflow(Qmin,Qmax,t0,t_end,t_systole,t_diastole)

# Define pressure profile ====================================#

# read pressure from csv file -> Comment out if not provided
# P = np.array(np.genfromtxt('Pressure_profile.csv'))
# Psys = np.max(P)
# Pdia = np.min(P)

#============================================================#

Psys = 30       # Systolic pressure
Pdia = 8        # Diastolic pressure
Pmean = (Psys+2*Pdia)/3     ## Mean pressure

Pdat = np.array([Psys, Pmean, Pdia])

# Define BC_matrix =========================================#

# BC_check = 1        ## if BC_check not 0, then code reverts to original windkessel BC calculation

# b1 = 10.0           #np.array([2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0])        ## R1
# b2 = 30.0           ## R2
# b3 = 0.10           ## C

# BC_vals = [b1,b2,b3]          # np.array([b1[i], b2, b3])

BC_scale = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0]       ## scale windkessel BC by the number used
# Call FSI code =============================================# 

for i in range(0,len(BC_scale)):

    wd = os.getcwd()+"\\"

    res_dir = wd+"\\data\\iter"+str(i)+"\\"

    iter_per = i*100.0/len(BC_scale)

    FSI_1D(wd, res_dir, Qin, tQ, kvals, Pdat, BC_scale[i],iter_per)
