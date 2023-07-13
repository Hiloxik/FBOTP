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
J1 = 0.8*np.pi
J2 = 0.9*np.pi
Jx = 1
m = 2
Jy_list = np.linspace(0,1,20)

d = 4

Nx = 10
Ny = 10
Nz = 10

N = 5

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
def Floquet(Jy):
    hopping_z1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z1[i,i+1] = J1
        hopping_z1[i+1,i] = J1
    for i in range(1,2*N-1,2):
        hopping_z1[i,i+1] = 0
        hopping_z1[i+1,i] = 0
    
    H1 = (3/T)*np.kron(np.eye(N),np.kron(np.eye(4*N), hopping_z1))
    
    hopping_z2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z2[i,i+1] = 0
        hopping_z2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_z2[i,i+1] = J2
        hopping_z2[i+1,i] = J2
    H2 = (3/T)*np.kron(np.eye(N),np.kron(np.eye(4*N), hopping_z2))
    
    hopping_z3 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z3[i,i] = 1
    for i in range(1,2*N,2):
        hopping_z3[i,i] = -1
    
    hopping_xy = np.zeros(((4*N)*(N),(4*N)*(N)),dtype=complex)
    for i in range(0,N):
        for j in range(i*(4*N)+1,i*(4*N)+4*N-2,4):
            hopping_xy[j,j+2] = m
            hopping_xy[j+2,j] = m
        for j in range(i*(4*N),i*(4*N)+4*N-2,4):
            hopping_xy[j,j+2] = m
            hopping_xy[j+2,j] = m
        for j in range(i*(4*N)+1,i*(4*N)+4*N-3,4):
            hopping_xy[j,j+2+4] = Jy
            hopping_xy[j+2+4,j] = Jy
        for j in range(i*(4*N)+2,i*(4*N)+4*N-3,4):
            hopping_xy[j,j+2] = Jx
            hopping_xy[j+2,j] = Jx
    for i in range(0,N-1):
        for j in range(i*(4*N)+2,i*(4*N)+4*N-1,4):
            hopping_xy[j,j+4*N-2] = Jy
            hopping_xy[j+4*N-2,j] = Jy
        for j in range(i*(4*N)+3,i*(4*N)+4*N,4):
            hopping_xy[j,j+4*N-2] = Jx
            hopping_xy[j+4*N-2,j] = Jx
    
    H3 = (3/T)*np.kron(hopping_xy,hopping_z3)

    UF = np.dot(expm(-1j*H1*(T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),expm(-1j*H1*(T/6))))))
    return UF

def main():
    fig = plt.figure(figsize=(7,7))
    # X, Z = np.meshgrid(np.linspace(0,2*N,2*N), np.linspace(0,2*N,2*N)) 
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_xlabel('x', fontsize=10, color='black')  
    # ax.set_ylabel('z', fontsize=10, color='black')  
    # ax.set_zlabel('E', fontsize=10, color='black')


    # for ky in ky_array:
    #     UF = Floquet(ky)
    #     eval,evec = np.linalg.eig(UF)
    #     Eval = []
    #     for j in range(len(eval)):
    #         Eval.append((1j/T)*llog(eval[j],np.pi))
    #     E = np.reshape(Eval,(2*N,2*N))
    #     ax.scatter3D(X, Z, E, color='blue')
    E = []
    for i in range(len(Jy_list)):
        Jy = Jy_list[i]
        UF = Floquet(Jy)
        eval,evec = np.linalg.eig(UF)
        Eval = []
        for j in range(len(eval)):
            Eval.append((1j/T)*llog(eval[j],np.pi))
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
