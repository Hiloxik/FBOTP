import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
from qutip import *
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D

#parameters
T = 1
J1 = 0.4*np.pi
J2 = 0.9*np.pi
phi = 0.2
m = 1

d = 4

Nx = 10
Ny = 10
Nz = 10

N = 10

kx_array = np.linspace(-np.pi,np.pi,2*N)
ky_array = np.linspace(-np.pi,np.pi,2*N)

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
    e = -1j*np.log(x)/T
    x_sort_idx = np.argsort(e)[::1]
    e = np.sort(e)[::1]
    y = y[:,x_sort_idx]
    x = e
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
def Floquet(kx, ky):
    hopping_z1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z1[i,i+1] = J1
        hopping_z1[i+1,i] = J1
    for i in range(1,2*N-1,2):
        hopping_z1[i,i+1] = 0
        hopping_z1[i+1,i] = 0
    H1 = (3/T)*np.kron(s0, hopping_z1)
    
    hopping_z2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z2[i,i+1] = 0
        hopping_z2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_z2[i,i+1] = J2
        hopping_z2[i+1,i] = J2
    H2 = (3/T)*np.kron(s0, hopping_z2)
    
    hopping_z3 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z3[i,i] = 1
    for i in range(1,2*N,2):
        hopping_z3[i,i] = -1
    H3 = (3/T)*phi*np.kron(np.sin(kx)*s1+np.sin(ky)*s2+(m+np.cos(kx)+np.cos(ky))*s3, hopping_z3)

    UF = np.dot(expm(-1j*H1*(T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),expm(-1j*H1*(T/6))))))
    return UF

def main():
    # states
    states = np.linspace(0,4*N,4*N)
    for kx in kx_array:
        for ky in ky_array:
            E = []
            UF = Floquet(kx,ky)
        
            eval,evec = np.linalg.eig(UF)
            for j in range(len(eval)):
                E.append((1j/T)*llog(eval[j],np.pi))
            E.sort()

        plt.scatter(states, E)
    plt.xticks(size=18,weight='bold')
    plt.yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    plt.xlabel(r'$states$',fontsize=20,fontweight='bold')
    plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    plt.axhline(0, color='black')
    plt.axhline(np.pi, color='black')
    plt.axhline(-np.pi, color='black')
    
    # bands
    # X, Y = np.meshgrid(np.linspace(-3.14,3.14,2*N), np.linspace(-3.14,3.14,2*N)) 
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # ax.set_yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # ax.set_zticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # ax.set_xlabel(r'$k_x$', fontsize=15, color='black',fontweight='bold')  
    # ax.set_ylabel(r'$k_y$', fontsize=15, color='black',fontweight='bold')  
    # ax.set_zlabel('E', fontsize=15, color='black',fontweight='bold')
    
    # E = []
    # for k in range(4*N):
    #     E.append(np.zeros((2*N,2*N)))
    # for i in range(len(kx_array)):
    #     kx = kx_array[i]
    #     for j in range(len(ky_array)):
    #         ky = ky_array[j]
    #         UF = Floquet(kx,ky)
    #         eval,evec = np.linalg.eig(UF)
    #         Eval = []
    #         for p in range(len(eval)):
    #             Eval.append((1j/T)*llog(eval[p],np.pi))
    #         Eval.sort()
    #         for l in range(len(Eval)):
    #             E[l][i,j] = Eval[l]
    # for w in range(4*N):
    #     ax.plot_surface(X, Y, E[w], cmap='viridis')
    
    # prob
    # K_array = np.array([-np.pi,np.pi])
    # for kx in K_array:
    #     for ky in K_array:
    #         UF = Floquet(kx,ky)
    #         eval,evec = np.linalg.eig(UF)
    #         eval,evec = ssort(eval,evec)
    #         edgewave = np.abs(evec[:,(eval>np.pi/T-0.05)&((eval<np.pi/T+0.05))])**2
    #         plt.plot(np.linspace(0,len(edgewave),len(edgewave)),edgewave)
    
    plt.show()
    
if __name__ == '__main__':
    main()     
