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
Theta = np.pi

#logarithm function with gap branch cut
def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi < argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j * argument

def U1(k):
    A = np.mat([[0,v],[v,0]])
    eiv1, ket1 = np.linalg.eig(A)
    Ueiv1 = np.exp(-1j*eiv1*t1)
    Udiag1 = np.diag(Ueiv1)
    U1 = np.dot(ket1,np.dot(Udiag1,np.conjugate(ket1).T))
    return U1

def U2(k):
    B = np.mat([[0,w*np.exp(-1j*k)],[w*np.exp(1j*k),0]])
    eiv2, ket2 = np.linalg.eig(B)
    Ueiv2 = np.exp(-1j*eiv2*t2)
    Udiag2 = np.diag(Ueiv2)
    U2 = np.dot(ket2,np.dot(Udiag2,np.conjugate(ket2).T))
    return U2

def Floquet(k):
    F = np.dot(U2(k),U1(k))
    eigenvalue, eigenvector = np.linalg.eig(F)
    Feiv = np.exp((-1/2)*np.log(eigenvalue))
    Fdiag = np.diag(Feiv)
    Floquet = np.dot(eigenvector,np.dot(Fdiag,np.conjugate(eigenvector).T))
    return Floquet

def U(k):  
    H1 = np.mat([[0,v],[v,0]])
    H2 = np.mat([[0,w*np.exp(-1j*k)],[w*np.exp(1j*k),0]])
    if t1 > t2:
        Eiv0, Ket0 = np.linalg.eig(H1)
        UEiv0 = np.exp(-1j*Eiv0*(T/2))
        UDiag0 = np.diag(UEiv0)
        U = np.dot(Ket0,np.dot(UDiag0,np.conjugate(Ket0).T))
    if t1 <= t2:
        Eiv1, Ket1 = np.linalg.eig(H1)
        UEiv1 = np.exp(-1j*Eiv1*(t1))
        UDiag1 = np.diag(UEiv1)
        U11 = np.dot(Ket1,np.dot(UDiag1,np.conjugate(Ket1).T))
        Eiv2, Ket2 = np.linalg.eig(H2)
        UEiv2 = np.exp(-1j*Eiv2*(T/2-t1))
        UDiag2 = np.diag(UEiv2)
        U22 = np.dot(Ket2,np.dot(UDiag2,np.conjugate(Ket2).T))
        U = np.dot(U22,U11)
    return U

def main():
    delta_1 = 1e-9 #differentiation step
    delta_2 = 1e-3 #integration step
    C=0
    for k in np.arange(-np.pi, np.pi, delta_2):
        AU0 = np.dot(U(k),Floquet(k))
        AU1 = np.dot(U(k+delta_1),Floquet(k+delta_1))
        C=C+np.trace(scipy.linalg.inv(AU0)*((AU1-AU0)/delta_1))*delta_2
    print('Winding number = ', (1j*C)/(np.pi),1)

if __name__ == '__main__':
    main()