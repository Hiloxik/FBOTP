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
J1 = 0.8*np.pi
J2 = 0.9*np.pi
Jx = 1
m = 2
Jy_list = np.linspace(0,1,20)

d = 4

Nx = 10
Ny = 10
Nz = 10

kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
kz_array = np.linspace(-np.pi,np.pi,Nz+1)

# print(ky_array)
t_array = np.linspace(7.5,12.5,21)
gap = np.pi
lx = len(kx_array)-1
ly = len(ky_array)-1
lz = len(kz_array)-1
lt = len(t_array)

#pauli matrix
s0 = np.eye(2)
s1 = sigmax()
s2 = sigmay()
s3 = sigmaz()
s4 = np.array([[0,-1j,0,0],[1j,0,0,0],[0,0,-1,0],[0,0,0,1]])

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
def Floquet(Jy,kx,ky,kz):
    #Hamiltonians
    H1 = (3/T)*J1*np.kron(np.kron(s1,s0),s0)
    H2 = (3/T)*J2*(np.cos(kz)*np.kron(np.kron(s1,s0),s0)+np.sin(kz)*np.kron(np.kron(s2,s0),s0))
    H3 = (3/T)*(-(Jx*np.sin(ky)+Jy*np.sin(kx))*np.kron(s3,s4)+(m+Jx*np.cos(ky)+Jy*np.cos(kx))*np.kron(np.kron(s3,s1),s0))
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),expm(-1j*H1*(T/6))))))
    
    return UF

def main():
    fig = plt.figure(figsize=(7,7))
    for kx in kx_array:
        for ky in ky_array:
            for kz in kz_array:
                E=[]
                for i in range(len(Jy_list)):
                    Jy = Jy_list[i]
                    UF = Floquet(Jy,kx,ky,kz)
                    eval,evec = np.linalg.eig(UF)
                    Eval = []
                    for k in range(len(eval)):
                        Eval.append((1j)*llog(eval[k],np.pi))
                    Eval.sort()
                    E.append(Eval)
                plt.plot(Jy_list,E,color='blue')
    plt.xticks(size=18,weight='bold')
    plt.yticks([-3.14,0,3.14],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    plt.xlabel(r'$m$',fontsize=20,fontweight='bold')
    plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(np.pi, color='black', linestyle='--')
    plt.axhline(-np.pi, color='black', linestyle='--')
    plt.show()
    
if __name__ == '__main__':
    main()    