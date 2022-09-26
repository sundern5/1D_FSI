from genericpath import exists
import os
import numpy as np
import matplotlib.pyplot as plt

def Plot_func(CFD_res_loc,Plot_loc, num_ves):

    fnames = os.listdir(CFD_res_loc)

    N = int(float(len(fnames))/num_ves)         ## Number of files per vessel

    for i in range(0,num_ves):

        t = np.zeros(N)

        P_prox = np.zeros(N)
        Q_prox = np.zeros(N)

        P_mid = np.zeros(N)
        Q_mid = np.zeros(N)

        P_dist = np.zeros(N)
        Q_dist = np.zeros(N)

        for j in range(0,N):
            file_name = CFD_res_loc + "\\Arteries_" + str(i+1) + "_" + str(j).zfill(3) + ".csv"
            if(exists(file_name)):
                file_dat = np.genfromtxt(file_name,dtype="float",delimiter = ",")

                num_pts = np.shape(file_dat)[0]

                t[j] = file_dat[0,0]            ## 1st column is time

                P_prox[j] = file_dat[0,2]           ## 3rd column is pressure rows are along length of tube
                P_mid[j] = file_dat[int(num_pts/2),2]
                P_dist[j] = file_dat[-1,2]

                Q_prox[j] = file_dat[0,3]           ## 4th column is flow rate
                Q_mid[j] = file_dat[int(num_pts/2),3]
                Q_dist[j] = file_dat[-1,3]
            else:
                continue

        var = np.array([t,P_prox,P_mid,P_dist,Q_prox,Q_mid,Q_dist])
        var = var.T
        dat_name = Plot_loc+'\\Artery_'+str(i+1)+'.csv'
        np.savetxt(dat_name,var, delimiter = ",")

        plt.rcParams.update({'font.size': 22})
        plt.subplot(1,2,1)
        plt.plot(t,P_prox,'-b')
        plt.plot(t,P_mid,'-r')
        plt.plot(t,P_dist, '-g')
        plt.rcParams.update({'font.size': 22})
        plt.xlabel("time")
        plt.ylabel("Pressure (mmHg)")
        plt.legend(['Proximal', 'Mid', 'Distal'])
        plt.rcParams.update({'font.size': 22})

        plt.subplot(1,2,2)
        plt.plot(t,Q_prox, '-b')
        plt.plot(t,Q_mid, '-r')
        plt.plot(t,Q_dist, '-g')
        plt.rcParams.update({'font.size': 22})
        plt.xlabel("time")
        plt.ylabel("Flow rate (ml/s)")
        plt.legend(['Proximal', 'Mid', 'Distal'])

        filename = Plot_loc+'\\Artery_'+str(i+1)+'.png'
        plt.savefig(filename)
        plt.close()
