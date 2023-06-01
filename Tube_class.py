from turtle import shape
import numpy as np
#import side_functions as sc

# Various constants ==============================#

cf = 1332.22                # Conversion factor from mmHg to g/cm/s^2
g = 981                     # Gravitational constant (g/cm^2)
ratio = 0.2                 # Windkessel Ratio (R1/RT) when not using Zc impedance
rho = 1.055                 # Density of blood, assumed constant (g/cm^3)
mu = 0.049                  # Viscosity of blood [g/cm/s].

tmstps = 20000                     # The number of timesteps per period.
plts   = 1024                     # Number of plots per period.
nu     = mu/rho                 # Dynamic viscosity of blood [cm^2/s].
Lr     = 1.0                    # Characteristic radius of the vessels in the tree [cm].
Lr2    = Lr**2                 # The squared radius [cm2].
Lr3    = Lr**3                 # The radius to the third power [cm^3].
q      = 10.0*Lr2               # The characteristic flow [cm^3/s].
Fr2    = (q**2)/(g*Lr**5)      # The squared Froudes number.
Re     = q*rho/(mu*Lr)            # Reynolds number.
p0     = 2.0*cf/(rho*g*Lr)      # Ensures a certain diastolic pressure.
Fcst = 1000

max_cycles = 40
cycles     = 1

t0 = 0.0        # initial time (s)
t_end = 0.85        # Duration of 1 cardiac cycle (s)

class artery:

    def __init__(self,L,r_top,r_bot,D1,D2,pts,Q_init,mech_props,BC):
        self.Length = L
        self.top_radius = r_top
        self.bottom_radius = r_bot
        self.branch1 = D1           # D1 will be an eger that hold the array location of the one of the branching tubes (Different from the C code)
        self.branch2 = D2
        self.grid_pts = pts         # number of grid pos per cm
        self.kvals = mech_props            # mech props will contain k1,k2,k3,k4 which are the properties of the tube wall 
        self.BC = BC            ## BC contains res1,res2 ,CT

        self.Q_init = Q_init

        self.N = int(self.grid_pts*self.Length)          # Get total number of pos of vessel
        
        self.r0 = np.zeros(self.N+1)
        self.r0h = np.zeros(self.N+2)
        self.dr0dx = np.zeros(self.N+1)
        self.dr0dxh = np.zeros(self.N+2)
        self.A0 = np.zeros(self.N+1)
        self.A0h = np.zeros(self.N+2)

        self.Qold = np.zeros(self.N+1)
        self.Aold = np.zeros(self.N+1)
        self.Qprv = np.zeros(self.N+1)
        self.Aprv = np.zeros(self.N+1)

        self.R1h = np.zeros(self.N)        
        self.R2h = np.zeros(self.N)
        self.S1h = np.zeros(self.N)
        self.S2h = np.zeros(self.N)   
        self.Ah = np.zeros(self.N)
        self.Qh = np.zeros(self.N)  

        self.qR=0.0
        self.aR=0.0
        self.cR=0.0
        self.HpR=0.0

        if self.Q_init==1:
            self.Q_read()

        self.calc_params()

    def calc_params(self):          # Split up the calculation part to a differnt function
        self.h = 1/(self.grid_pts*Lr)

        # rgLr  = 4.0/3.0/rho/g/Lr
        # rgLr2 = 4.0/3.0/rho/g/Lr2

        for i in range(0,self.N+2):             ## I think the h suffixes are half steps
            if i!=self.N+1:
                self.r0[i] = self.top_radius*np.exp(i*np.log(self.bottom_radius/self.top_radius)/self.N)/Lr     ## radius at length l, vessel is assumed to taper exponentially        
                self.r0h[i] = self.top_radius*np.exp((i-0.5)*np.log(self.bottom_radius/self.top_radius)/self.N)/Lr
            
                self.dr0dx[i] = np.log(self.bottom_radius/self.top_radius)/self.h/self.N*self.r0[i]            ## derivative of r0 ad r0h wrt l
                self.dr0dxh[i] = np.log(self.bottom_radius/self.top_radius)/self.h/self.N*self.r0h[i] 
            
                self.A0[i] = np.pi*((self.r0[i])**2)          ## cross-sectional area at length l
                self.A0h[i] = np.pi*((self.r0h[i])**2)
            
            elif i==self.N+1:
                self.r0h[i] = self.top_radius*np.exp((i-0.5)*np.log(self.bottom_radius/self.top_radius)/self.N)/Lr
                self.dr0dxh[i] = np.log(self.bottom_radius/self.top_radius)/self.h/self.N*self.r0h[i] 
                self.A0h[i] = np.pi*(self.r0h[i])**2

        self.Qnew = np.zeros(self.N+1)
        self.Anew = self.A0.copy()

    def Q_read(self):
        self.Q0 = np.array(np.genfromtxt('flow_profile.csv', dtype= 'float', delimiter = ',', usecols=0))
        self.Q0 = self.Q0/q

    def F(self,Q,A):
        var = -(Fcst*np.pi*Q)/(A*Re)  
        return var  

    def dFdQ(self, A):
        return(-Fcst*np.pi/(A*Re))

    def dFdA(self,Q,A):
        return(Fcst*np.pi*Q/((A**2)*Re)) 

# When determining or checking the step-size (k) the CFL-condition is applied.
# This is determined according to the result reached from the analysis
# made using the method of characteristics (See IMFUFATEKST no 297).
# In this function the minimal step-size fulfilling this condition for this
# tube is returned.

    def CFL(self):
        minimum = 64000000.0
        for i in range(0,self.N+1):
            c_tmp = np.sqrt(0.5*75.0*np.sqrt(self.Anew[i]/self.A0[i])/Fr2)
            Vnew = self.Qnew[i]/self.Anew[i]
            temp = np.min([self.h/np.abs(Vnew - c_tmp), self.h / np.abs(Vnew + c_tmp)])
            if (temp < minimum):
                minimum = temp
        return minimum    

    def P(self, i, A):
        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/(self.A0[i]**2))*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        # if(A/self.A0[i]>2):
        #     print(str(alpha)+'_'+str(A/self.A0[i])+'_'+str(self.Length))
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0[i])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        pold = self.kvals[0]*(1-self.A0[i]/A)+alpha*beta*gamma
        #print("P="+str(pold)) 
        return pold

    def dPdA(self,i,A):
        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0[i])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        trm1 = self.kvals[0]*self.A0[i]/(A**2)  
        trm2 = alpha*beta*gamma*(4.0*self.kvals[2]*((A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)\
                                       *((A/self.A0[i]**2)*(np.cos(self.kvals[3]))**2))
        trm3 = -alpha*gamma*((1/self.A0[i])*(np.cos(self.kvals[3]))**2)
        trm4 = alpha*beta*((2*A/self.A0[i])*(np.cos(self.kvals[3]))**2)

        pold = trm1+trm2+trm3+trm4
        #print("dPdA="+str(pold)) 
        return pold

    def dPdx1(self,i,A):
        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0[i])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        trm1 = -(2.0*self.kvals[0]*np.pi*np.sqrt(self.A0[i])/A)*self.dr0dx[i] 
        trm2 = -alpha*beta*gamma*(8.0*self.kvals[2]*((A**2/self.A0[i]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)\
                                       *((A**2*np.sqrt(np.pi)/self.A0[i]**(2.5))*(np.cos(self.kvals[3]))**2))
        trm3 = +alpha*gamma*((2*A*np.sqrt(np.pi)/self.A0[i]**(1.5))*(np.cos(self.kvals[3]))**2)
        trm4 = -alpha*beta*((4*A**2*np.sqrt(np.pi)/self.A0[i]**(2.5))*(np.cos(self.kvals[3]))**2)

        pold = trm1+trm2+trm3+trm4
        #print("dPdx1="+str(pold)) 
        return pold

    def B(self,i,A):

        trm1 = A*self.P(i,A)

        trm2 = -(self.kvals[0]*np.sqrt(np.pi)/rho)*self.A0[i]#*(A-self.A0[i]+np.log(self.A0[i]/A))

        trm3 = 0

        if(A!=self.A0[i]):
            a = np.linspace(self.A0[i],A,num=21)
            dA = a[1]-a[0]

            for j in range(0,21):
                trm3 -= self.P(i,a[j])*dA

        pold = trm1+trm2+trm3
        #print("B="+str(pold)) 
        return pold

    def Bh(self,i,A):
        ip1 = i+1

        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/(self.A0h[ip1]**2))*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0h[ip1])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        trm1 = A*self.kvals[0]*(1-self.A0h[ip1]/A)+alpha*beta*gamma 

        trm2 = -(self.kvals[0]*np.sqrt(np.pi)/rho)*self.A0h[ip1]#*(A-self.A0h[ip1]+np.log(self.A0h[ip1]/A))

        trm3 = 0
        
        if(A!=self.A0h[ip1]):
            a = np.linspace(self.A0h[ip1],A,num=21)
            dA = a[1]-a[0]
            for j in range(0,21):
                trm3 -= self.P(i,a[j])*dA

        pold = trm1+trm2+trm3
        #print("Bh="+str(pold)) 
        return pold

    def dBdx1(self,i,A):

        pold = (A/rho)*self.dPdx1(i,A)
        #print("dBdx1="+str(pold)) 

        return pold

    def dBdx1h(self,i,A):
        ip1 = i+1

        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0h[ip1])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        trm1 = -(2.0*self.kvals[0]*np.pi*np.sqrt(self.A0h[ip1])/A)*self.dr0dxh[ip1] 
        trm2 = -alpha*beta*gamma*(8.0*self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)\
                                       *((A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2))
        trm3 = +alpha*gamma*((2*A*np.sqrt(np.pi)/self.A0h[ip1]**(1.5))*(np.cos(self.kvals[3]))**2)
        trm4 = -alpha*beta*((4*A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2)

        pold = (A/rho)*(trm1+trm2+trm3+trm4)
        #print("dBdx1h="+str(pold)) 
        return pold

    def dBdAh(self,i,A):
        ip1 = i+1

        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0h[ip1])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        trm1 = self.kvals[0]*self.A0h[ip1]/(A**2)  
        trm2 = alpha*beta*gamma*(4.0*self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)\
                                       *((A/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2))
        trm3 = -alpha*gamma*((1/self.A0h[ip1])*(np.cos(self.kvals[3]))**2)
        trm4 = alpha*beta*((2*A/self.A0h[ip1])*(np.cos(self.kvals[3]))**2)

        pold = (A/rho)*(trm1+trm2+trm3+trm4)
        #print("dBdAh="+str(pold)) 
        return pold

    def d2BdAdxh(self,i,A):
        ip1 = i+1

        alpha = (2.0*self.kvals[1]/3.0)*np.exp(self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)**2)
        beta = 5.0*(np.sin(self.kvals[3]))**2-(A/self.A0h[ip1])*(np.cos(self.kvals[3]))**2
        gamma = (A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2

        delta = (8.0*self.kvals[2]*((A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+(np.sin(self.kvals[3]))**2-1)\
                                       *((A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2))

        trm1 = -(2.0*self.kvals[0]*np.pi*np.sqrt(self.A0h[ip1])/A)
        trm2 = -alpha*beta*gamma*delta
        trm3 = +alpha*gamma*((2*A*np.sqrt(np.pi)/self.A0h[ip1]**(1.5))*(np.cos(self.kvals[3]))**2)
        trm4 = -alpha*beta*((4*A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2)

        trm11 = (2.0*self.kvals[0]*np.pi*np.sqrt(self.A0h[ip1])/A**2)

        trm21 = -alpha*beta*gamma*(8.0*self.kvals[2]*((4.0*A**2/self.A0h[ip1]**2)*(np.cos(self.kvals[3]))**2+2.0*(np.sin(self.kvals[3]))**2)\
                                       *((A*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2))
        trm22 = +alpha*gamma*delta*((np.sqrt(np.pi)/self.A0h[ip1])*(np.cos(self.kvals[3]))**2)
        trm23 = -alpha*beta*delta*((2.0*A*np.sqrt(np.pi)/self.A0h[ip1])*(np.cos(self.kvals[3]))**2)
        trm24 = -alpha*beta*gamma*delta**2
        
        trm31 = +alpha*gamma*((2.0*A*np.sqrt(np.pi)/self.A0h[ip1]**(1.5))*(np.cos(self.kvals[3]))**2)*delta        
        trm32 = +alpha*gamma*((2*np.sqrt(np.pi)/self.A0h[ip1]**(1.5))*(np.cos(self.kvals[3]))**2)
        trm33 = +alpha*((2.0*A*np.sqrt(np.pi)/self.A0h[ip1]**(1.5))*(np.cos(self.kvals[3]))**2)\
                *((2.0*A/self.A0h[ip1]**(2))*(np.cos(self.kvals[3]))**2)

        trm41 = -alpha*beta*((4*A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2)*delta
        trm42 = -alpha*beta*((8*A*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2)
        trm43 = +alpha*((4*A**2*np.sqrt(np.pi)/self.A0h[ip1]**(2.5))*(np.cos(self.kvals[3]))**2)*((np.sqrt(np.pi)/self.A0h[ip1])*(np.cos(self.kvals[3]))**2)

        dPdx1h = (1/rho)*(trm1+trm2+trm3+trm4)

        d2PdAdx = (A/rho)*(trm11+trm21+trm22+trm23+trm24+trm31+trm32+trm33+trm41+trm42+trm43)

        pold = (dPdx1h+d2PdAdx)
        # print("d2BdAdxh="+str(pold)) 
        return pold

# When taking a Lax-Wendroff step, the flux of the system must be determined.
# This is evaluated at i + j/2, and the prediction is given as described
# in IMFUFATEKST no 297 and D2.1-4. The eger k determines whether we deal
# with the first or the second component of the vector.

    def Rvec(self,k,i,j,Q,A):
        if(k==1):
            return Q
        elif(k==2):
            var = (Q**2)/A + (self.B(i,A) if j==0 else self.Bh(i,A))
            return var
        else: 
            print("arteries.cxx","Call of non-existing vector-component of R")
            return 0

# Similarly the right hand side of the system of equations must be determined
# at i + j/2. Also in this case the function is given as stated in
# the mathematical model, and also in this case k states the needed component
# of the vector.

    def Svec(self,k,i,j,Q,A):
        if(k==1):
            return 0.0
        elif(k==2):
            var = self.F(Q,A) + (self.dBdx1(i,A) if j==0 else self.dBdx1h(i,A))
            #print([self.F(Q,A), (self.dBdx1(i,A) if j==0 else self.dBdx1h(i,A))])
            return var
        else: 
            print("arteries.cxx","Call of non-existing vector-component of S")
            exit()

# The solutions of Anew and Qnew are found for all interior points
# of the vessel at (t+k), where k is the length of the current
# time-step. This function saves the results in the arrays Anew and
# Qnew, and the function is made according to Lax-Wendroff's method
# as described in Olufsen, et al., Ann Biomed Eng 28, 1281?1299, 2000.            

    def step(self,k):
        theta = k/self.h
        gamma = 0.5*k

        self.Qold = self.Qnew.copy()
        self.Aold = self.Anew.copy()

        R1 = np.zeros(self.N+1)
        S1 = np.zeros(self.N+1) 
        R2 = np.zeros(self.N+1) 
        S2 = np.zeros(self.N+1) 

        for i in range(0,self.N+1):
            R1[i] = self.Rvec(1,i,0,self.Qold[i],self.Aold[i])
            R2[i] = self.Rvec(2,i,0,self.Qold[i],self.Aold[i])
            S1[i] = self.Svec(1,i,0,self.Qold[i],self.Aold[i])
            S2[i] = self.Svec(2,i,0,self.Qold[i],self.Aold[i])

        for i in range(0,self.N):
            self.Ah[i] = 0.5*(self.Aold[i+1]+self.Aold[i]) - 0.5*theta*(R1[i+1]-R1[i]) + 0.5*gamma*(S1[i+1]+S1[i])
            self.Qh[i] = 0.5*(self.Qold[i+1]+self.Qold[i]) - 0.5*theta*(R2[i+1]-R2[i]) + 0.5*gamma*(S2[i+1]+S2[i])
            self.R1h[i] = self.Rvec(1,i,1,self.Qh[i],self.Ah[i])
            self.R2h[i] = self.Rvec(2,i,1,self.Qh[i],self.Ah[i])
            self.S1h[i] = self.Svec(1,i,1,self.Qh[i],self.Ah[i])
            self.S2h[i] = self.Svec(2,i,1,self.Qh[i],self.Ah[i])

        #print(S2)    

        for i in range(1,self.N):
            self.Anew[i] = self.Aold[i] - theta*(self.R1h[i]-self.R1h[i-1]) + gamma*(self.S1h[i]+self.S1h[i-1])
            self.Qnew[i] = self.Qold[i] - theta*(self.R2h[i]-self.R2h[i-1]) + gamma*(self.S2h[i]+self.S2h[i-1])

        # if(np.max(self.Anew)/np.max(self.A0)>2):
        #     print(str(self.Anew))
        

# The left boundary (x=0) uses this function to model an inflow into
# the system. The actual parameter given to the function is the model time.
# As stated in the mathematical model the constants of the function are
# chosen in order to ensure a certain CO (specified in main.h). Hence we have
# the specified value of b. Further the period (dimension-less) is assumed
# to be Period.
    def Q0_init (self,t,k,Period):
        if (t <= Period):
            return self.Q0[int(np.interp(t,np.array([0,Period]),np.array([0,np.size(self.Q0)-1])))]
        elif (t >  Period): 
            return (self.Q0_init((t-Period),k,Period))
        else:
            return (0)

# The value at the right boundary at time t is predicted. NB: This should
# only be used with terminal vessels, i.e. for vessels that don't bifurcate
# into further branches.
# In that situation the bifurcation boundary function should be called
# instead. Again the procedure specified is given according to the mathemati-
# cal theory presented in Olufsen, et al., Ann Biomed Eng 28, 1281?1299, 2000.

    def c(self, i, A):          # The wave speed through aorta.
        cnst = np.sqrt(0.5*75.0*np.sqrt(A/self.A0[i])/Fr2)  #Linear B
        return cnst

    def Hp (self,i,Q,A):
        var = (self.F(Q,A) - A*self.dPdx1(i,A)/Fr2)/(-Q/A + self.c(i,A))
        return var

    def Hn (self,i,Q,A):
        var = (self.F(Q,A) - A*self.dPdx1(i,A)/Fr2)/(-Q/A - self.c(i,A))
        return var

# Update of the left boundary at time t. This function uses Q0 to determine
# the flow rate at the next time-step. From this the value of A is predicted
# using Lax-Wendroff's numerical scheme. This function is only relevant
# when the tube is an inlet vessel.
    def negchar (self,theta):
        ctm1 = self.c(0, self.Aold[0])
        Hntm1 = self.Hn(0, self.Qold[0], self.Aold[0])
        uS = self.Qold[0]/self.Aold[0]
        ch = (uS - ctm1) * theta
    
        if (ctm1 - uS < 0):
            print("ctm1 - uS < 0, CFL condition violated\n")
            exit(1)
    
        self.qS = self.Qold[0] + (self.Qold[0] - self.Qold[1])*ch
        self.aS = self.Aold[0] + (self.Aold[0] - self.Aold[1])*ch
        self.cS = ctm1 + (ctm1  - self.c (1,self.Aold[1]))*ch
        self.HnS = Hntm1 + (Hntm1 - self.Hn(1,self.Qold[1],self.Aold[1]))*ch

    def bound_left (self,t,k,Period):
        
        self.Qnew[0] = self.Q0_init(t,k,Period)

        if ((t/k) < 0):
            print("t/k negative in bound_left\n")
    
        self.negchar(k/self.h)
        uS = self.qS/self.aS
        self.Anew[0] = self.aS + (self.Qnew[0] - self.qS)/(uS + self.cS) + k*self.HnS

    def poschar (self,theta):
        ctm1  = self.c(self.N, self.Aold[self.N])
        Hptm1 = self.Hp(self.N, self.Qold[self.N], self.Aold[self.N])
        uR = self.Qold[self.N] / self.Aold[self.N]
        ch = (uR + ctm1) * theta

        if (uR + ctm1 < 0):
            print("uR + ctm1 < 0, CFL condition violated\n")
            exit(1)

        self.qR  = self.Qold[self.N] - (self.Qold[self.N] - self.Qold[self.N-1])*ch
        self.aR  = self.Aold[self.N] - (self.Aold[self.N] - self.Aold[self.N-1])*ch
        self.cR  = ctm1 - (ctm1  - self.c(self.N-1,self.Aold[self.N-1]))*ch
        self.HpR = Hptm1 - (Hptm1 - self.Hp(self.N-1,self.Qold[self.N-1],self.Aold[self.N-1]))*ch

    def bound_right(self,k,theta,t):
        j = 1
        ok = 'false'
        ntrial = 60
    
        self.poschar(theta)#, self.qR, self.aR, self.cR, self.HpR)
        
        uR = self.qR/self.aR
        k1 = 1/(1 + k*(self.BC[0]+self.BC[1])/(self.BC[0]*self.BC[1]*self.BC[2]))
        k2 = k/(self.BC[0]*self.BC[1]*self.BC[2])
        cst = (k1*(self.Qold[self.N]-self.P(self.N,self.Aold[self.N])/self.BC[0]) - self.qR)/(self.cR-uR) - self.aR - self.HpR*k
    
    # Initial guesses
    
        xr = self.Anew[self.N-1]
        f = 0
        df = 0
        
        while (j <= ntrial and ok=='false'):
            f  = xr + cst + k1*(1/self.BC[0] +k2)*self.P(self.N,xr)/(self.cR-uR)
            df = 1 + k1*(1/self.BC[0]+k2)*self.dPdA(self.N,xr)/(self.cR-uR)
            ch, xr = zero_1d(xr, f, df, 1.0e-4)
            if (xr <= 0.0):
                print("WARNING (arteries.C): Bound_right: x was negative")
                xr = self.Anew[self.N-1] # Bound xr[1] away from zero.
            if (ch == 1):
                ok = 'true'
            j = j+1
        # Solutions are applied, and right boundary and the intermediate array QL
        # are updated.
        self.Anew[self.N] = xr
        self.Qnew[self.N] = k1*(self.Qold[self.N] + (self.P(self.N,self.Anew[self.N])-self.P(self.N,self.Aold[self.N]))/self.BC[0] + k2*self.P(self.N,self.Anew[self.N]))
        #print(self.Qold[self.N])
        
        # If the solution is not found print an error message. We don't use
        # subroutine error,
        # since it can't take the function values as arguments.
        if (j >= ntrial):
            print("WARNING (arteries.C): Root not found in the right boundary")
            self.Anew[self.N] = self.Ah[self.N-1]
            self.Qnew[self.N] = self.Qh[self.N-1]

# Takes one step with Newton Raphson's method. Assumes to find zeros for
# a one-dimensional problem.

def zero_1d(x,f,df,tolx):
  dx = f/df
  x = x-dx
  #if (fabs(dx) < tolx) return(1) else   // Original statement.
  var = ((np.abs(dx) < tolx) and (np.fabs(f) < tolx))
  #print(np.abs(dx),np.fabs(f))
  return var, x   