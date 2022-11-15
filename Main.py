#============================================================#
#                                                            #
# Main Code                                                  #
# Version 1.0                                                #
# Ported by Sunder                                           #
# Description: This is the code to run for the 1D FSI        #
# problem. This code is a python version of the MJ colebank  #
# code written in MATLAB and C.                              #
#                                                            #
#============================================================#

# Import header functions======================================#

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Import side functions======================================#

import side_functions as ic
from Tube_class import artery

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

#######################################################################################
######                        START OF MAIN function                             ######
#######################################################################################

def FSI_1D(wd, res_dir, Qin, tQ, kvals, Pdat, BC_scale, iter_per):

# Set working directories======================================#

    CFD_res_loc = res_dir       ## location where results are being stored

    if (os.path.exists(CFD_res_loc)==False):            ## Check for existence of results directory (if true -> clear directory, else -> create the directory)
        os.makedirs(CFD_res_loc)
    else:
        filelist = glob.glob(os.path.join(CFD_res_loc, "*"))
        for f1 in filelist:
            os.remove(f1)

    Plot_loc = wd+ "\\figures"            ## location for plots to be stored    

    if (os.path.exists(Plot_loc)==False):            ## Check for existence of results directory (if true -> clear directory, else -> create the directory)
        os.makedirs(Plot_loc)
    else:
        filelist = glob.glob(os.path.join(Plot_loc, "*"))
        for o in filelist:
            os.remove(o)

# Specify vessel dimensions ================================#

# Store length, inner and our radii in vessel_dims.csv with columns in that order

    L = np.array(np.genfromtxt(wd+'Vessel_dims.csv',dtype= 'float', delimiter = ',', usecols=0))      
     
    Rout = np.array(np.genfromtxt(wd+'Vessel_dims.csv', dtype= 'float', delimiter = ',', usecols=2))

    dims = np.array(np.genfromtxt(wd+'Vessel_dims.csv', dtype= 'float', delimiter = ',')) 

# Get connectivity matrix ===================================#

    connectivity = np.array([np.genfromtxt(wd+'Connectivity.csv', dtype= 'int', delimiter = ',')])  

    terminal_vessel = np.array(np.genfromtxt(wd+'Terminal_vessels.csv', dtype= 'int', delimiter = ',')) 

    num_ves = np.size(L)
    num_term = np.size(terminal_vessel)
    num_pts  = 8

# Specify Windkessel conditions ==============================#

    IMP_FLAG = 0        ## Set to one if you want to use "characterisitc impedance" -> not fully sure what this means

    # if(BC_check!=0):
    #     BC_matrix = ic.windkessel(IMP_FLAG,connectivity,terminal_vessel,np.flip(L),np.flip(Rout),Qin,Pdat,kvals,tQ)
    # else:
    #     BC_matrix = np.ones((num_ves,3))
    #     BC_matrix[:,0] *= BC_vals[0]
    #     BC_matrix[:,1] *= BC_vals[1]
    #     BC_matrix[:,2] *= BC_vals[2]
    
    BC_matrix = ic.windkessel(IMP_FLAG,connectivity,terminal_vessel,np.flip(L),np.flip(Rout),Qin,Pdat,kvals,tQ)
    BC_matrix[:,0] *= BC_scale
    BC_matrix[:,1] *= BC_scale
    BC_matrix[:,2] *= 1.0/BC_scale


# Initialize vessels as objects ==============================#    
    total_conn = num_ves-num_term    
    conn_id = 0
    term_id = 0#num_term-1
    bc_id = 0#num_term-1

    Arteries = []

    if (num_ves==1):
        var = artery(dims[0,0],dims[0,1],dims[0,2],0,0,num_pts,1,kvals,BC_matrix[0,:])
        Arteries.append(var)
    elif (num_ves>1):
        for i in range(0,num_ves):
            if(conn_id<total_conn):
                if(i==connectivity[0,conn_id,0]-1):
                    curr_d1 = connectivity[0,conn_id,1]
                    curr_d2 = connectivity[0,conn_id,2]
                    conn_id = conn_id+1

            if(i==0):
                var = artery(dims[i,0],dims[i,1],dims[i,2],curr_d1-1,curr_d2-1,num_pts,1,kvals,np.array([0,0,0]))
                Arteries.append(var)  
            else:
                if (term_id<num_term and (i==terminal_vessel[term_id]-1)):
                    var =  artery(dims[i,0],dims[i,1],dims[i,2],0,0,num_pts,0,kvals,BC_matrix[bc_id,:])
                    Arteries.append(var)
                    term_id = term_id+1
                    bc_id = bc_id+1
                else:
                    var =  artery(dims[i,0],dims[i,1],dims[i,2],curr_d1-1,curr_d2-1,num_pts,0,kvals,np.array([0,0,0]))
                    Arteries.append(var)  

# Start solving time steps ==============================#     
    
    t0 = 0.0        # initial time (s)
    t_end = 1      # Duration of 1 cardiac cycle (s)
    
    period = t_end*q/Lr3        # The dimension-less period.
    k      = period/tmstps      # Length of a timestep.
    Deltat = period/plts        # Interval between each point plotted.
    
    iter = 1
    max_iter = 5

    tstart = 0.0
    tend = Deltat

    tol = 1e-8
    er = 100.0            ## arbitrary value greater than tol

    ctr = 1               ## Check counter to only write certain iterations

    n_ctr = 2             ## write every n-th step       

    while (tend<=period):

        iter = 1

        while(iter<=max_iter): ## and er>tol
            ic.solver(tstart,tend,k,period,Arteries,num_ves)
            
            if(iter==1):
                P_er1 = ((Arteries[0].P(0,Arteries[0].Anew[0]))**2 + (Arteries[0].P(-1,Arteries[0].Anew[-1]))**2)*(rho*g*Lr/cf)**2 
                er = np.abs(P_er1)
            else:
                P_er2 = ((Arteries[0].P(0,Arteries[0].Anew[0]))**2 + (Arteries[0].P(-1,Arteries[0].Anew[-1]))**2)*(rho*g*Lr/cf)**2
                er = np.abs(P_er1-P_er2) 
                P_er1 = P_er2

            if(ctr%n_ctr==0):
                for i in range(0,1):
                    A1 = np.zeros(Arteries[i].N+1)
                    A2 = np.zeros(Arteries[i].N+1)
                    A3 = np.zeros(Arteries[i].N+1)
                    A4 = np.zeros(Arteries[i].N+1)
                    A5 = np.zeros(Arteries[i].N+1)
                    A6 = np.zeros(Arteries[i].N+1)

                    for j in range(0,Arteries[i].N+1):
                        A1[j] = tend*Lr3/q
                        A2[j] = Lr*j*Arteries[i].h
                        A3[j] = (Arteries[i].P(j,Arteries[i].Anew[j]))*rho*g*Lr/cf+12
                        A4[j] = q*Arteries[i].Qnew[j]
                        A5[j] = Lr2*Arteries[i].Anew[j]
                        A6[j] = Arteries[i].c(j,Arteries[i].Anew[j])*Fr2

                    var = np.array([A1,A2,A3,A4,A5,A6])
                    var = var.T

                    fname = CFD_res_loc + "Arteries_" + str(i+1) + "_" + str(int(tstart*100)).zfill(4) + ".csv"

                    np.savetxt(fname,var, delimiter = ",")
            
            iter +=1

        tstart = tend   
        tend = tend+Deltat   
        ctr = ctr+1 

        percent_comp = 100.0*tstart/period
        os.system('cls')
        print(str(iter_per)+"__"+str(percent_comp))

    ic.Plot_func(CFD_res_loc,Plot_loc,1)
    os.system('cls')
               