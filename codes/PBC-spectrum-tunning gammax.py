from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
from qutip import *
import pandas as pd
import time

#parameters
T = 1
J12 = (np.pi)/(np.sqrt(2))
J11_list = np.linspace(0,J12,20)

J2 = (np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
Nx = 20
Ny = 20
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(7.5,12.5,21)
gap = np.pi
lx = len(kx_array)-1
ly = len(ky_array)-1
lt = len(t_array)

#pauli matrix
s0 = np.eye(2)
s1 = sigmax()
s2 = sigmay()
s3 = sigmaz()

#generalized logarithm
def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j*argument

#states-sorting function
def ssort(x,y):
    e = -1j*np.log(x)
    x_sort_idx = np.argsort(e)[::-1]
    e = np.sort(e)[::-1]
    xx = []
    for i in range(len(x)):
        xx.append(np.exp(1j*e[i]))
    y = y[:,x_sort_idx]
    x = xx
    return x,y

def SSort(x,y):
    e = []
    for i in range(len(x)):
        e.append(((1j/(2*np.pi))*llog(x[i],0)).real)
    E = np.array(e)
    x_sort_idx = np.argsort(E)[::-1]
    x = x[x_sort_idx]
    y = y[:,x_sort_idx]
    return x,y

#phase-fixing function
def fix(x,d):
    Phase = np.zeros(shape=(d,d),dtype=complex)
    for j in range(d):
        Phase[j,j] = np.exp(-1j*cmath.phase(x[0,j]))
    x = np.dot(x,Phase)
    return x

#Floquet operator
def Floquet(J11,kx,ky):
    #Hamiltonians
    H1 = (2/T)*(J12*np.kron(s1,s0)-J11*np.kron(s2,s2))+(0*2/T)*(np.sin(ny*ky)*np.kron(s3,s1)+np.cos(ny*ky)*np.kron(s3,s2)-0.4*np.cos(ny*ky)*np.kron(s0,s0))+(0.0001)*(np.kron(s3,s0)+0.5*np.kron(s3,s3))
    H2 = (2*J2/T)*((np.cos(nx*kx))*np.kron(s1,s0)-(np.sin(nx*kx))*np.kron(s2,s3)-(np.cos(ny*ky))*np.kron(s2,s2)-(np.sin(ny*ky))*np.kron(s2,s1))
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/4)),np.dot(expm(-1j*H2*(T/2)),expm(-1j*H1*(T/4))))
    
    return UF

def main():
    for kx in kx_array:
        for ky in ky_array:
            E1 = []
            E2 = []
            for i in range(len(J11_list)):
                J11 = J11_list[i]
                UF = Floquet(J11,kx,ky)
            
                eval,evec = np.linalg.eig(UF)
                Eval = []
                for j in range(len(eval)):
                    Eval.append((1j/T)*llog(eval[j],0))
                Eval.sort()

                E1.append(Eval[0])
                E2.append(Eval[3])
            plt.plot(J11_list,E1,color='blue')
            plt.plot(J11_list,E2,color='blue')
    pi_list = []
    for i in range(len(J11_list)):
        pi_list.append(np.pi)
    plt.plot(J11_list,pi_list, color='black', linestyle='--')
    plt.xticks(size=18,weight='bold')
    plt.yticks(size=18,weight='bold')
    plt.xlabel(r'$\gamma_x$',fontsize=20,fontweight='bold')
    plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    plt.axvline(J12, color='black', linestyle='--')   
    plt.show()
    
if __name__ == '__main__':
    main()     