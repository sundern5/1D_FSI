import numpy as np
import matplotlib.pyplot as plt
import os

wd = "D:\\Blender_project\\1D_FSI\\Python_port\\Data_res\\"

os.chdir(wd)

files = os.listdir()

N = len(files)

X = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]) 
X = X/10.0

P = np.zeros(N)

for i in range(0,N):

    os.chdir(wd+files[i])

    fnames = os.listdir()

    num_files = len(fnames)

    t = np.zeros(num_files)

    P_prox = np.zeros(num_files)

    for j in range(0,num_files):
        file_name = "Arteries_1_" + fnames[j][-8:-4] + ".csv" 
        file_dat = np.genfromtxt(file_name,dtype="float",delimiter = ",") 

        t[j] = file_dat[0,0]            ## 1st column is time
        P_prox[j] = file_dat[0,2]           ## 3rd column is pressure rows are along length of tu

    P[i] = np.amax(P_prox)

    os.chdir(wd)

plt.figure(figsize=(16, 10), dpi=150)
plt.rcParams.update({'font.size': 22})
plt.plot(X,P,'-b')
plt.rcParams.update({'font.size': 22})
plt.xlabel("Resistance ratio")
plt.ylabel("Max Pressure (mmHg)")
plt.rcParams.update({'font.size': 22})

filename = 'Resistance_variation.png'
plt.savefig(filename)
plt.close()