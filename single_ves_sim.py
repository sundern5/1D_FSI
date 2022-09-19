import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import time, sys
#============================================================#
#                                                            #
# Main Code                                                  #
# Version 1.0                                                #
# Ported by Sunder                                           #
# Description: This is the code to run for the 1D FSI        #
# problem. THis code is a python version of the MJ colebank  #
# code written in MATLAB and C.                              #
#                                                            #
#============================================================#

# Import side functions======================================#

import side_functions as ic
from Tube_class import artery

working_loc = "D:\\Blender_project\\1D_FSI\Python_port\\single_ves\\"              ## location where files to read are stored

os.chdir(working_loc)

CFD_res_loc = "D:\\Blender_project\\1D_FSI\Python_port\\single_ves\\data"        ## location where results are being stored


filelist = glob.glob(os.path.join(CFD_res_loc, "*"))
for f in filelist:
    os.remove(f)

Plot_loc = "D:\\Blender_project\\1D_FSI\Python_port\\single_ves\\figures"            ## location for plots to be stored    

filelist = glob.glob(os.path.join(Plot_loc, "*"))
for f in filelist:
    os.remove(f)

    # Various constants ==============================#

cf = 1332.22                # Conversion factor from mmHg to g/cm/s^2
g = 981.0                     # Gravitational constant (g/cm^2)
ratio = 0.2                 # Windkessel Ratio (R1/RT) when not using Zc impedance
rho = 1.055                 # Density of blood, assumed constant (g/cm^3)
mu = 0.049                  # Viscosity of blood [g/cm/s].

tmstps = 20000               # The number of timesteps per period.
plts   = 1024                     # Number of plots per period.
nu     = mu/rho                 # Dynamic viscosity of blood [cm^2/s].
Lr     = 1.0                    # Characteristic radius of the
Lr2    = Lr**2                 # The squared radius [cm2].
Lr3    = Lr**3                 # The radius to the third power [cm^3].
q      = 10.0*Lr2               # The characteristic flow [cm^3/s].
Fr2    = (q**2)/(g*Lr**5)      # The squared Froudes number.
Re     = q*rho/(mu*Lr)            # Reynolds number.
p0     = 2.0*cf/(rho*g*Lr)      # Ensures a certain diastolic pressure.

max_cycles = 40
cycles = 1

# The Solver function ==============================#

# k is the time step at each iteration

def solver(t_start,t_end,k,period,artery,num_ves):

    t = t_start
    qLnb = (t/k)%tmstps

    while t<t_end:

        if(t+k>t_end):          # Check if stepping through time takes it past the end time       
            k = t_end-t         # update the step time to mathc the time bounds

        for i in range(0,num_ves):
            if(k>artery[i].CFL()):
                print(t)
                print(k)
                print(artery[i].CFL())
                print("Step size too large, exiting\n")
                exit() 

        for i in range(0,num_ves):          
            artery[i].step(k)
                   
        artery[0].bound_left(t+k,k,period)
        #print(Arteries[0].Anew[0]/Arteries[0].Anew[0])

        f = open("Q0.txt", "a")
        f.write(str(t+k) + "," + str(q*artery[0].Qnew[0]) + "\n")
        f.close()


        for i in range(0,num_ves):
            if(artery[i].branch1==0):
                artery[i].bound_right(k,k/artery[i].h,t)
               
            else:
                theta = k/artery[i].h
                gamma = k/2
                ic.bound_bif(artery,i,theta,gamma)


        t=t+k
        qLnb = (qLnb+1)%tmstps


#######################################################################################
######                        START OF MAIN SCRIPT                               ######
#######################################################################################

# Define vessel mechanical properties =======================# !! NEED TO MODIFY THIS !!

# Eh/r0 = k1*exp(k2*r0) + k3
# where E, h, and r0 are the Youngs Modulus, wall thickness, and reference radius, respectively.

k1 = 0
k2 = -25
k3 = 9e+4

# Define flow waveform ======================================#

# read flow rate from csv file -> Comment out if not provided
# Qin = np.array(np.genfromtxt('flow_profile.csv', usecols=1))
# tQ = np.array(np.genfromtxt('flow_profile.csv', usecols=0))

#============================================================#

Qmin = 0        # Min flow rate (ml/s)
Qmax = 50      # Max flow rate (ml/s)

t0 = 0.0        # initial time (s)
t_end = 1      # Duration of 1 cardiac cycle (s)

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

# Specify vessel dimensions ================================#

L = np.array([10.0])

Rin = np.array([1.0])
Rout = np.array([1.0])

terminal_vessel = np.array([1]) 

BC_matrix =np.array([[20.0,60.0,0.2]])

num_pts  = 5

period = t_end*q/Lr3        # The dimension-less period.
k      = period/tmstps      # Length of a timestep.
Deltat = period/plts        # Interval between each point plotted.

tstart    = 0.0         # start time

kvals = [k1, k2, k3]       ## supposed elastic parameters for function

Arteries = []

var = artery(L,Rin,Rout,0,0,num_pts,1,kvals,BC_matrix[0,:])

Arteries.append(var)

iter = 1
max_iter = 5

tstart = 0.0
tend = Deltat

tol = 1e-8
er = 100.0            ## arbitrary value greater than tol

f = open("test.txt", "a")

while (tend<=period):

    iter = 1

    while(er>tol and iter<max_iter):#period_counter*period):
        
        #print([tstart,iter])

        solver(tstart,tend,k,period,Arteries,1)

        if(iter==1):
            P_er1 = ((Arteries[0].P(0,Arteries[0].Anew[0]))**2 + (Arteries[0].P(-1,Arteries[0].Anew[-1]))**2)*(rho*g*Lr/cf)**2 
            er = np.abs(P_er1)
            #print(er)
        else:
            P_er2 = ((Arteries[0].P(0,Arteries[0].Anew[0]))**2 + (Arteries[0].P(-1,Arteries[0].Anew[-1]))**2)*(rho*g*Lr/cf)**2
            er = np.abs(P_er1-P_er2) 
            P_er1 = P_er2

        for i in range(0,1):
            A1 = np.zeros(Arteries[i].N+1)
            A2 = np.zeros(Arteries[i].N+1)
            A3 = np.zeros(Arteries[i].N+1)
            A4 = np.zeros(Arteries[i].N+1)
            A5 = np.zeros(Arteries[i].N+1)
            A6 = np.zeros(Arteries[i].N+1)
            A7 = np.zeros(Arteries[i].N+1)

            for j in range(0,Arteries[i].N+1):
                A1[j] = tend*Lr3/q
                A2[j] = Lr*j*Arteries[i].h
                A3[j] = (Arteries[i].P(j,Arteries[i].Anew[j]))*rho*g*Lr/cf
                A4[j] = q*Arteries[i].Qnew[j]
                A5[j] = Lr2*Arteries[i].Anew[j]
                A6[j] = Arteries[i].c(j,Arteries[i].Anew[j])*Fr2
                A7[j] = Lr2*Arteries[i].A0[j]

            var = np.array([A1,A2,A3,A4,A5,A6,A7])
            var = var.T

            fname = CFD_res_loc + "\\Arteries_" + str(i+1) + "_" + str(int(tstart*100)).zfill(3) + ".csv"

            np.savetxt(fname,var, delimiter = ",")
        
        iter +=1
        if(iter>max_iter):
            f.write(str(tstart) + "\n")

    tstart = tend    
    tend = tend+Deltat    

    percent_comp = 100.0*tstart/period
    os.system('cls')
    print(percent_comp)


ic.Plot_func(CFD_res_loc,Plot_loc, 1, iter)
f.close()
os.system('cls')