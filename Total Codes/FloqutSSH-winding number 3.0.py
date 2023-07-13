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
aalpha=0.25 #phase index 1
bbeta=1.25 #phase index 2
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
def U(k):  
    matrix = np.zeros((2, 2), dtype=complex)
    if t1 >= t2:
        matrix[0,1] = v*(T/2)
        matrix[1,0] = v*(T/2)
    if t1 < t2:
        matrix[0,1] = v*t1+w*(T/2-t1)*cmath.exp(-1j*k)
        matrix[1,0] = v*t1+w*(T/2-t1)*cmath.exp(1j*k)
    return scipy.linalg.expm(-1j*matrix)

#Floquet operator cut at half period
def UF(k):
    E=np.sqrt(math.pow(v*t1,2)+math.pow(w*t2,2)+2*(v*t1)*(w*t2)*np.cos(k))
    eiphi=((v*t1)+(w*t2)*np.exp(1j*k))/(E)
    eiphic=np.conjugate(eiphi)
    TT=(1/np.sqrt(2))*np.matrix([[eiphic,eiphic],[1,-1]])
    Tc=(1/np.sqrt(2))*np.matrix([[eiphi,1],[eiphi,-1]])
    EIG=np.exp(-1j*E)
    EIGc=np.exp(1j*E)
    hamiltonian=np.matrix([[(1j/T)*llog(EIG,Theta),0],[0,(1j/T)*llog(EIGc,Theta)]])
    Hamiltonian = TT*hamiltonian*Tc #effective Hamiltonian
    return scipy.linalg.expm(1j*(T/2)*Hamiltonian)

#chiral matrix
def S(k):
    diag,P=np.linalg.eig(U(k)*UF(k)) #eigen-problem of anamolous time operator at half period
    Trans0=np.matrix([[1,0],[0,1]])
    Trans1=np.matrix([[1,0],[0,-1]])
    Trans2=(1/np.sqrt(2))*np.matrix([[1,1],[1,-1]])
    if Theta == np.pi:
        S=P*Trans1*scipy.linalg.inv(P)
    if Theta == 0:
        S=P*Trans2*Trans1*Trans2*scipy.linalg.inv(P)
    return np.round((1/2)*(Trans0-S),2)

#calculate the winding number
def main():
    start_clock = time.perf_counter()
    delta_1 = 1e-5 #derivation step
    delta_2 = 1e-3 #integration step
    W = 0  
    for k in np.arange(-np.pi, np.pi, delta_2):
        H0 = U(k)*UF(k)
        H1 = U(k+delta_1)*UF(k+delta_1)
        W = W + np.trace(S(k)*scipy.linalg.inv(H0)*((H1-H0)/delta_1))*delta_2 # Winding number
    print('Winding number = ', (1j*W)/(2*np.pi))
    end_clock = time.perf_counter()

if __name__ == '__main__':
    main()