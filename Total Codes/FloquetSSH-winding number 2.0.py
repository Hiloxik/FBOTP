import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

#parameter settings
v=1 #intracell hopping in time period 1
w=1 #intercell hopping in time period 2
aalpha=-0.25 #phase index 1
bbeta=0.75 #phase index 2
t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
print(t1,t2)
T=t1+t2 #total time period
Theta=np.pi #gap

#logarithm function with gap branch cut
def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j * argument

#time evolution operator at half period
def U(k,t):  
    matrix1 = np.zeros((2, 2), dtype=complex)
    matrix2 = np.zeros((2, 2), dtype=complex)
    if t < t1:
        matrix1[0,1] = v*(t)
        matrix1[1,0] = v*(t)
        matrix2[0,1] = 0
        matrix2[1,0] = 0
    if t >= t1:
        matrix1[0,1] = v*t1
        matrix1[1,0] = v*t1
        matrix2[0,1] = w*(t-t1)*cmath.exp(-1j*k)
        matrix2[1,0] = w*(t-t1)*cmath.exp(1j*k)
    return np.dot(scipy.linalg.expm(-1j*matrix2),scipy.linalg.expm(-1j*matrix1))

#Floquet operator cut at half period
def UF(k,t):
    E=np.sqrt(math.pow(v*t1,2)+math.pow(w*t2,2)+2*(v*t1)*(w*t2)*np.cos(k))
    eiphi=((v*t1)+(w*t2)*np.exp(1j*k))/(E)
    eiphic=np.conjugate(eiphi)
    TT=(1/np.sqrt(2))*np.matrix([[eiphic,eiphic],[1,-1]])
    Tc=(1/np.sqrt(2))*np.matrix([[eiphi,1],[eiphi,-1]])
    EIG=np.exp(-1j*E)
    EIGc=np.exp(1j*E)
    hamiltonian=np.matrix([[(1j/T)*llog(EIG,Theta),0],[0,(1j/T)*llog(EIGc,Theta)]])
    Hamiltonian = TT*hamiltonian*Tc #effective Hamiltonian
    return scipy.linalg.expm(1j*(t/T)*Hamiltonian)

#calculate the winding number
def main():
    delta_1 = 1e-5 #differentiation step
    delta_2 = 1e-3 #integration step
    for t in np.arange(0,T,delta_2):
        for k in np.arange(-np.pi, np.pi, delta_2):
            H0 = U(k,t)*UF(k,t)
            H1 = U(k+delta_1,t)*UF(k+delta_1,t)
            C = np.trace(scipy.linalg.inv(H0)*((H1-H0)/delta_1))*delta_2
        W = C*delta_2   
    print('Winding number = ', (1j*C)/(2*np.pi))

if __name__ == '__main__':
    main()