from cmath import exp
from genericpath import exists
import os
import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
from Tube_class import artery


# Various constants ==============================#

cf = 1332.22                # Conversion factor from mmHg to g/cm/s^2
g = 981                     # Gravitational constant (g/cm^2)
ratio = 0.2                 # Windkessel Ratio (R1/RT) when not using Zc impedance
rho = 1.055                 # Density of blood, assumed constant (g/cm^3)
mu = 0.049                  # Viscosity of blood [g/cm/s].

tmstps = 20000                   # The number of timesteps per period.
plts   = 1024                    # Number of plots per period.
nu     = mu/rho                 # Dynamic viscosity of blood [cm^2/s].
Lr     = 1.0                    # Characteristic radius of the
Lr2    = Lr**2                 # The squared radius [cm2].
Lr3    = Lr**3                 # The radius to the third power [cm^3].
q      = 10.0*Lr2               # The characteristic flow [cm^3/s].
Fr2    = (q**2)/(g*Lr**5)      # The squared Froudes number.
Re     = q*rho/(mu*Lr),            # Reynolds number.
p0     = 2.0*cf/(rho*g*Lr)      # Ensures a certain diastolic pressure.
EPS = 1.0e-20

# Function to get flow profile ==============================#

def inflow(Qmin,Qmax,t0,t_end,t_systole,t_diastole):
  N = tmstps        ## data points in inflow

  delta_t = (t_end-t0)/N
  t = np.linspace(t0,t_end,N+1)

  Q = np.zeros(N+1)

  b = t_diastole/2

  c = 0.05

  Q = Qmin+(Qmax-Qmin)*(np.exp(-((t-b)**2)/(2*c**2)))

  # for i in range(0,N+1):
  #   ti = i*delta_t
  #   if(t0<ti and ti<=t_systole):
  #       Q[i] = 0.5*(Qmax - Qmin)*(1 - np.cos(np.pi*ti/t_systole)) + Qmin
  #   elif(ti>t_systole and ti<=t_diastole):
  #       Q[i] = 0.5*(Qmax - Qmin)*(1 + np.cos(np.pi*(ti-t_systole)/(t_diastole-t_systole))) + Qmin
  #   elif(ti>t_diastole and ti<=t_end):
  #       Q[i] = Qmin

  #var = np.array([Q,t])
  np.savetxt("flow_profile.csv",Q,delimiter=",")
  return Q, t        

# Function to get windkessel parameters ==============================#

def windkessel(IMP_FLAG,connectivity,terminal_vessel,L,Rout,Qin,Pdat,kvals,tQ):
  num_vessels = np.size(L)

  r41     = np.zeros(num_vessels)
  Q       = np.zeros(num_vessels)
  Rtotal  = np.zeros(num_vessels)
  R1      = np.zeros(num_vessels)
  R2      = np.zeros(num_vessels)
  CT      = np.zeros(num_vessels)
  Zc      = np.zeros(num_vessels)

  # Non-dimensionalize ================================#

  P_mean = Pdat[1]
  tau = tau_minimal_data(Pdat,Qin,tQ)
  
  # Nondimensional all constants
  Q_spread = np.mean(Qin)
  Lr = 1.0                    # Nondimensional length
  qc = 10*Lr**2                # Nondimensional flow
  Pnd = P_mean*(cf/(rho*g*Lr))  # Nondimensional mean pressure
  Qnd = Q_spread/qc           # Nondimensional mean flow
  tc = tau*qc/Lr**3            # Nondimensional tau parameter for WK

  stiff = lambda r : (kvals[0]*np.exp(kvals[1]*r) + kvals[2])*(4/3)*(rho/(g*Lr))

  r41 = (Rout**4)/L

  conn_inc = 0

  a = np.array(connectivity[0,:,0])
  #print(a)
  #print(connectivity[:,0])
  for i in a:         # This distributes the flow during branching based on resistance
    if i==1:
      Q[i-1] = Qnd
    if(any(a==i)):  
      d1 = connectivity[:,conn_inc,1]
      d2 = connectivity[:,conn_inc,2]
      Q[d1-1] = Q[i-1]*r41[d1-1]/(r41[d1-1] + r41[d2-1])
      Q[d2-1] = Q[i-1]*r41[d2-1]/(r41[d1-1] + r41[d2-1])
      conn_inc += 1

  for i in range(0,num_vessels):
    Rtotal[i] = Pnd/Q[i]            # total resistance from P/Q over time
    R1[i] = round(ratio*Rtotal[i],4)
    R2[i] = round(Rtotal[i]-R1[i],4)
    CT[i] = tc/Rtotal[i]
    Zc[i] = (1/(np.pi*Rout[i]**2))*np.sqrt(rho*stiff(Rout[i])/2)          # impedance

  param = np.zeros(num_vessels*2 + np.size(terminal_vessel)*3)
  inneri=0
  for i in range(0,2*num_vessels,2):
    param[i] = L[inneri]
    param[i+1] = Rout[inneri]
    inneri = inneri + 1

  BC_matrix = np.zeros((np.size(terminal_vessel),3))

# Decide whether to use alpha/(1-alpha) rule or use Zc.
  if IMP_FLAG == 1:
    R1 = Zc
    R2 = Rtotal - Zc

  for inneri in range(0,np.size(terminal_vessel)):#(2*num_vessels+1,np.size(param),3):
    if(round(CT[terminal_vessel[inneri]-1],6)==0):
      BC_matrix[inneri,:] = [round(R1[terminal_vessel[inneri]-1],2),round(R2[terminal_vessel[inneri]-1],2),1e-6]
    else:
      BC_matrix[inneri,:] = [round(R1[terminal_vessel[inneri]-1],2),round(R2[terminal_vessel[inneri]-1],2),round(CT[terminal_vessel[inneri]-1],6)]
  
  return BC_matrix

# Sub function for Windkessel function==============================#

def tau_minimal_data(p,q,t):
  N = np.size(t)

  dqdt = np.gradient(q,t)     ## time derivative of flow rate
  
  nc = 3

  n = 0.0

  m = np.floor((N-1)/nc)
  id = np.array(np.where(dqdt<=1e-3))
  #print(id)

  N1 = np.size(id)

  idl = np.zeros(N1)

  for i in range(0,N1):
    idtest = id[0,i]
    if(idtest>m):
        idl[i] = idtest

  idl = idl[idl!=0]
  ids = int(idl[0]-n) 
  
  p0 = p[1]
  t0 = np.array([t[ids]])

  p1 = p[2]      # get pressure and time range for diastole
  t1 = np.array(t[-1])

  #print(np.array([t0,t1,p0,p1]))

  fun = lambda x : (p0 - p0)**2 + (p1 - p0*np.exp(-(t1-t0)/x))**2     # Optimize this function to get tau

  xopt = sc.fmin(func=fun, x0=0.1,maxiter=15000,disp='False')

  tau=xopt

  return tau

# Functions in tools.c==============================#

def lubksb(a,n,indx,b):

  ii = 0

  for i in range(0,n):
    ip = indx[i]
    var = b[int(ip-1)]
    b[int(ip-1)] = b[i]

    if (ii != 0):
      for j in range(ii-1,i):
        var = var - a[i,j]*b[j]
    elif (var):
      ii=i+1
    b[i] = var
    
  for i in range(n-1,-1,-1):
    var1 = b[i]

    for j in range(i+1,n):
      var1 = var1 - a[i,j]*b[j]

    b[i] = var1 / a[i,i]

  return b


def ludcmp(a,n,indx,d):

  imax = -1
  vv=np.zeros(n)
  d = 1.0

  for i in range(0,n):
    big = 0.0
    for j in range(0,n):
      big = big if big>np.abs(a[i,j]) else np.abs(a[i,j])
    if(big==0.0):
      print("Singular matrix in routine LUDCMP")
      print("Numerical Recipes run-time error...\n")
      print("...now exiting to system...\n")
      exit()
    vv[i] = 1.0/big

  for j in range(0,n):
    for i in range(0,j):
      var = a[i,j]
      for k in range(0,i):
        var = var - a[i,k]*a[k,j]
      a[i,j] = var

    big = 0.0
    
    for i in range(j,n):
      var1 = a[i,j]
      for k in range(0,j):
        var1 = var1 - a[i,k]*a[k,j]
      a[i,j] = var1
      dum = vv[i]*np.abs(var1)
      if(dum >= big):
        big = dum
        imax = i

    if (j!=imax) :
      for k in range(0,n):
        tmp = a[imax,k]
        a[imax,k] = a[j,k]
        a[j,k] = tmp
      d = -d
      vv[imax] = vv[j]

    indx[j] = imax+1
    
    if (a[j,j] == 0.0):
       a[j,j] = EPS
    if (j+1 != n):
      dum = 1.0/(a[j,j])
      for i in range(j+1,n):
        a[i,j] = a[i,j]*dum
  return a

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

        # f = open("Q0.txt", "a")
        # f.write(str(t+k) + "," + str(q*artery[0].Qnew[0]) + "\n")
        # f.close()

        for i in range(0,num_ves):
            if(artery[i].branch1==0):
                artery[i].bound_right(k,k/artery[i].h,t)
               
            else:
                theta = k/artery[i].h
                gamma = k/2
                bound_bif(artery,i,theta,gamma)

        t=t+k
        qLnb = (qLnb+1)%tmstps

# Takes one step with Newton Raphson's method. Assumes to find zeros
# for a multi-dimensional problem.
def zero(x,n,tolx,tolf,fvec,fjac):

  indx=np.zeros(n)
  p=np.zeros(n)

  errf = np.sum(np.abs(fvec))

  if (errf <= tolf):
    return 1, x

  p = -fvec  
  d = 1.0

  fjac = ludcmp(fjac, n, indx, d)
  p = lubksb(fjac, n, indx, p)

  errx = 0.0
  for i in range(0,n):
    errx = errx + np.abs(p[i])
    x[i] = x[i] + p[i]

  if(errx <= tolx):
    return 1, x
  else:
    return 2, x


# Takes one step with Newton Raphson's method. Assumes to find zeros for
# a one-dimensional problem.
def zero_1d(x,f,df,tolx):
  dx  = f/df
  x  = x-dx
  #if (fabs(dx) < tolx) return(1) else   // Original statement.
  var = (np.abs(dx) < tolx & np.fabs(f) < tolx)
  return var


# The value at the bifurcation point at time t is predicted. This should
# only be done for tubes that do bifurcate into further branches. If
# this is not the case we have a terminal vessel and bound_right should be
# called instead. The procedure operates according to the specifications
# in the mathematical model as a link between this tube and its daughters.
# Therefore there will be three tubes involved in this function.
# One problem is however, that the rather complicated system of equations does
# not converge for all choices of parameters (the peripheral resistance, the
# top radius, and the bottom radius).

def bound_bif(artery,i,theta,gamma):
  j = 1
  ok = 'false'
  ntrial = 40

  Art = artery[i]
  LD = artery[Art.branch1]
  RD = artery[Art.branch2]

  g1 = Art.Qold[Art.N] + theta*Art.R2h[Art.N-1] + gamma*Art.S2h[Art.N-1]
  g2 = LD.Qold[0] - theta*(LD.R2h[0]) + gamma*(LD.S2h[0])
  g2 = LD.Qold[0] - theta*(LD.R2h[0]) + gamma*(LD.S2h[0])
  g2a = RD.Qold[0] - theta*(RD.R2h[0]) + gamma*(RD.S2h[0])

  k1   = Art.Aold[Art.N] + theta*Art.R1h[Art.N-1]
  k2   = LD.Aold[0] - theta*(LD.R1h[0])
  k2a  = RD.Aold[0] - theta*(RD.R1h[0])

  k3   = Art.Qh[Art.N-1]/2.0
  k4   = LD.Qh[0]/2.0
  k4a  = RD.Qh[0]/2.0

  k5   = Art.Ah[Art.N-1]/2.0
  k6   = LD.Ah[0]/2.0
  k6a  = RD.Ah[0]/2.0

  #print("done1")

  xb=np.zeros(18)

  # The approximative initial guesses are applied.
  xb[0] =  Art.Qh[Art.N-1]                                                           #Initial guess for Q1_xb n+1
  xb[1] = (Art.Qold[Art.N-1] + Art.Qold[Art.N])/2.0                      #Initial guess for Q1_xb^n+0.5
  xb[2] =  Art.Qold[Art.N]                                                           #Initial guess for Q1_xb+0.5 n+0.5
  xb[3] =  LD.Qh[0]                                                       #Initial guess for Q2_xb n+1
  xb[4] = (LD.Qold[0] + LD.Qold[1])/2.0            #Initial guess for Q2_xb n+0.5
  xb[5] =  LD.Qold[0]                                                     #Initial guess for Q2_xb+0.5 n+0.5
  xb[6] =  RD.Qh[0]                                                       #Initial guess for Q3_xb n+1
  xb[7] = (RD.Qold[0] + RD.Qold[1])/2.0            #Initial guess for Q3_xb n+0.5
  xb[8] =  RD.Qold[0]                                                     #Initial guess for Q3_xb+0.5 n+0.5
  xb[9] =  Art.Ah[Art.N-1]                                                           #Initial guess for A1_xb n+1
  xb[10] = (Art.Aold[Art.N-1] + Art.Aold[Art.N])/2.0                     #Initial guess for A1_xb^n+0.5
  xb[11] =  Art.Aold[Art.N]                                                          #Initial guess for A1_xb+0.5 n+0.5
  xb[12] =  LD.Ah[0]                                                      #Initial guess for A2_xb n+1
  xb[13] = (LD.Aold[0] + LD.Aold[1])/2.0           #Initial guess for A2_xb n+0.5
  xb[14] =  LD.Aold[0]                                                    #Initial guess for A2_xb+0.5 n+0.5
  xb[15] =  RD.Ah[0]                                                      #Initial guess for A3_xb n+1
  xb[16] = (RD.Aold[0] + RD.Aold[1])/2.0           #Initial guess for A3_xb n+0.5
  xb[17] =  RD.Aold[0]                                                    #Initial guess for A3_xb+0.5 n+0.5

  #print(xb)
  k7nh  = 0 
  k7n   = 0 
  k7anh = 0 
  k7an  = 0 

  #print("done2")

  # The residuals (fvec), and the Jacobian is determined, and if possible
  # the system of equations is solved.
  while(j <= ntrial and ok=='false'): # Find the zero

      fvec = np.zeros(18)
      # The residuals.
      fvec[0] = g1 - xb[0] - theta*((xb[2]**2)/xb[11] + Art.Bh(Art.N,xb[11])) + gamma*(Art.F(xb[2],xb[11])+Art.dBdx1h(Art.N,xb[11]))

      fvec[1] = g2 - xb[3] + theta*((xb[5]**2)/xb[14] + LD.Bh(-1,xb[14])) + gamma*(Art.F(xb[5],xb[14])  + LD.dBdx1h(-1,xb[14]))

      fvec[2] = g2a - xb[6] + theta*((xb[8]**2)/xb[17] + RD.Bh(-1,xb[17])) + gamma*(Art.F(xb[8],xb[17])  + RD.dBdx1h(-1,xb[17]))

      fvec[3] = -theta*xb[2] - xb[9]  + k1
      fvec[4] = theta*xb[5] - xb[12]  + k2
      fvec[5] = theta*xb[8] - xb[15]  + k2a
      fvec[6] = -xb[1] + xb[2]/2.0 + k3
      fvec[7] = -xb[4] + xb[5]/2.0 + k4
      fvec[8] = -xb[7] + xb[8]/2.0 + k4a
      fvec[9] = -xb[10] + xb[11]/2.0 + k5
      fvec[10] = -xb[13] + xb[14]/2.0 + k6
      fvec[11] = -xb[16] + xb[17]/2.0 + k6a
      fvec[12] = -xb[1] + xb[4] + xb[7]
      fvec[13] = -xb[0] + xb[3] + xb[6]

      PN = Art.P(Art.N,xb[10])
      sq211 = (xb[1]/xb[10])**2

      if (xb[1] > 0):
          fvec[14] = -PN + LD.P(0,xb[13]) + k7nh*sq211
          fvec[15] = -PN + RD.P(0,xb[16]) + k7anh*sq211
      else:
          fvec[14] = -PN + LD.P(0,xb[13]) - k7nh*sq211
          fvec[15] = -PN + RD.P(0,xb[16]) - k7anh*sq211

      PN  = Art.P(Art.N,xb[9])
      sq110 = (xb[0]/xb[9])**2

      if (xb[0] > 0):
          fvec[16] = -PN + LD.P(0,xb[12]) + k7n*sq110
          fvec[17] = -PN + RD.P(0,xb[15]) + k7an*sq110
      else:
          fvec[16] = -PN + LD.P(0,xb[12]) - k7n*sq110
          fvec[17] = -PN + RD.P(0,xb[15]) - k7an*sq110

      fjac = np.zeros((18,18))   

      # The Jacobian.
      fjac[0,0]  = -1.0
      fjac[13,0]  = -1.0

      if (xb[0] > 0):
          fjac[16,0] = xb[0]/(xb[9]**2)*(2*k7n)#-1)
          fjac[17,0] = xb[0]/(xb[9]**2)*(2*k7an)#-1)
      else:
          fjac[16,0] = xb[0]/(xb[9]**2)*(-2*k7n)#-1)
          fjac[17,0] = xb[0]/(xb[9]**2)*(-2*k7an)#-1)

      fjac[6,1] = -1.0
      fjac[12,1] = -1.0

      if (xb[1] > 0):
          fjac[14,1] = xb[1]/(xb[10]**2)*(2*k7nh)#-1)
          fjac[15,1] = xb[1]/(xb[10]**2)*(2*k7anh)#-1)
      else:
          fjac[14,1] = xb[1]/(xb[10]**2)*(-2*k7nh)#-1)
          fjac[15,1] = xb[1]/(xb[10]**2)*(-2*k7anh)#-1)

      fjac[0,2] = -2.0*theta*xb[2]/xb[11] + gamma*Art.dFdQ(xb[11])

      fjac[3,2] = -theta
      fjac[6,2] = 0.5

      fjac[1,3] = -1.0
      fjac[13,3] = 1.0

      fjac[7,4] = -1.0
      fjac[12,4] = 1.0

      fjac[1,5] = 2.0*theta*xb[5]/xb[14] + gamma*Art.dFdQ(xb[14])

      fjac[4,5] = theta
      fjac[7,5] = 0.5

      fjac[2,6] = -1.0
      fjac[13,6] = 1.0

      fjac[8,7] = -1.0
      fjac[12,7] = 1.0

      fjac[2,8] = 2.0*theta*xb[8]/xb[17] + gamma*Art.dFdQ(xb[17])

      fjac[5,8] = theta
      fjac[8,8] = 0.5

      fjac[3,9] = -1.0

      if (xb[0] > 0):
          fjac[16,9] = -Art.dPdA(Art.N,xb[9]) + (xb[0]**2)/(xb[9]**3)*(-2.0*k7n)#+1)
          fjac[17,9] = -Art.dPdA(Art.N,xb[9]) + (xb[0]**2)/(xb[9]**3)*(-2.0*k7an)#+1)
      else:
          fjac[16,9] = -Art.dPdA(Art.N,xb[9]) + (xb[0]**2)/(xb[9]**3)*(2.0*k7n)#+1)
          fjac[17,9] = -Art.dPdA(Art.N,xb[9]) + (xb[0]**2)/(xb[9]**3)*(2.0*k7an)#+1)

      fjac[9,10] = -1.0

      if (xb[1] > 0):
          fjac[14,10] = -Art.dPdA(Art.N,xb[10]) + (xb[1]**2)/(xb[10]**3)*(-2.0*k7nh)#+1)
          fjac[15,10] = -Art.dPdA(Art.N,xb[10]) + (xb[1]**2)/(xb[10]**3)*(-2.0*k7anh)#+1)
      else:
          fjac[14,10] = -Art.dPdA(Art.N,xb[10]) + (xb[1]**2)/(xb[10]**3)*(2.0*k7nh)#+1)
          fjac[15,10] = -Art.dPdA(Art.N,xb[10]) + (xb[1]**2)/(xb[10]**3)*(2.0*k7anh)#+1)

      fjac[0,11] = theta*((xb[2]/xb[11])**2 - Art.dBdAh(Art.N,xb[11])) + gamma*(Art.dFdA(xb[2],xb[11]) + Art.d2BdAdxh(Art.N,xb[11]))
      fjac[9,11] = 0.5

      fjac[4,12] = -1.0
      fjac[16,12] = LD.dPdA(0,xb[12])

      fjac[10,13] = -1.0
      fjac[14,13] = LD.dPdA(0,xb[13])

      fjac[1,14] = theta*(-(xb[5]/xb[14])**2 + LD.dBdAh(-1,xb[14])) + gamma*(Art.dFdA(xb[5],xb[14]) + LD.d2BdAdxh(-1,xb[14]))
      fjac[10,14] = 0.5

      fjac[5,15] = -1.0
      fjac[17,15] = RD.dPdA(0,xb[15])

      fjac[11,16] = -1.0
      fjac[15,16] = RD.dPdA(0,xb[16])

      fjac[2,17] = theta*(-((xb[8]/xb[17])**2) + RD.dBdAh(-1,xb[17])) + gamma*(Art.dFdA(xb[8],xb[17]) + RD.d2BdAdxh(-1,xb[17]))
      fjac[11,17] = 0.5

      # Check whether solution is close enough. If not run the loop again.
      ch, XB = zero(xb, 18, 1.0e-12, 1.0e-12, fvec, fjac)
      if(ch == 1):
          ok = 'true'

      xb = XB

      j = j+1

  # Solutions is applied, and right boundary is updated.
  Art.Anew[Art.N] = xb[9]
  Art.Qnew[Art.N] = xb[0]
  LD.Anew[0] = xb[12]
  LD.Qnew[0] = xb[3]
  RD.Anew[0] = xb[15]
  RD.Qnew[0] = xb[6]

  artery[i] = Art
  artery[artery[i].branch1] = LD
  artery[artery[i].branch2] = RD

  if (j >=ntrial):
      print("arteries.C","Root not found in the bifurcation")
      exit()

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
          file_name = CFD_res_loc + "\\Arteries_" + str(i+1) + "_" + fnames[j][-8:-4] + ".csv"
          file_dat = np.genfromtxt(file_name,dtype="float",delimiter = ",")

          num_pts = np.shape(file_dat)[0]

          t[j] = file_dat[0,0]            ## 1st column is time

          P_prox[j] = file_dat[0,2]           ## 3rd column is pressure rows are along length of tube
          P_mid[j] = file_dat[int(num_pts/2),2]
          P_dist[j] = file_dat[-1,2]

          Q_prox[j] = file_dat[0,3]           ## 4th column is flow rate
          Q_mid[j] = file_dat[int(num_pts/2),3]
          Q_dist[j] = file_dat[-1,3]

      var = np.array([t,P_prox,P_mid,P_dist,Q_prox,Q_mid,Q_dist])
      var = var.T
      dat_name = Plot_loc+'\\Artery_'+str(i+1)+'.csv'
      np.savetxt(dat_name,var, delimiter = ",")
      
      fig, axs = plt.subplots(2, 2, figsize=(20, 10))

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
