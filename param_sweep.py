import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import side_functions as ic
from Tube_class import artery
from Main import FSI_1D

# Define vessel mechanical properties =======================# 

k1 = 40
k2 = 5.0
k3 = 0.08         
k4 = 55.0*np.pi/180.0

#kvals = [k1, k2, k3, k4]       ## ontained from fitting

# Define flow waveform ======================================#

# read flow rate from csv file -> Comment out if not provided
# Qin = np.array(np.genfromtxt('flow_profile.csv', usecols=1))
# tQ = np.array(np.genfromtxt('flow_profile.csv', usecols=0))

#============================================================#

Qmin = 20        # Min flow rate (ml/s)
Qmax = 20    # Max flow rate (ml/s)

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

## scaling for resistance for PH -> 8

BC_scale = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0,20.0,30.0,40.0,50.0]       ## scale windkessel BC by the number used
# Call FSI code =============================================# 

for i in range(0,len(BC_scale)):

    wd = os.getcwd()+"\\retool\\resistance_variation\\"

    res_dir = wd+"\\data\\iter_"+str(i).zfill(2)+"\\"

    iter_per = 100.0*i/len(BC_scale)

    kvals = [k1, k2, k3, k4]

    FSI_1D(wd, res_dir, Qin, tQ, kvals, Pdat, BC_scale[i],iter_per)
