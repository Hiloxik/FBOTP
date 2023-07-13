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
Jx = 1
Jy = 1
m = 0.5
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
def Floquet(ky):
    hopping_z1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z1[i,i+1] = J1
        hopping_z1[i+1,i] = J1
    for i in range(1,2*N-1,2):
        hopping_z1[i,i+1] = 0
        hopping_z1[i+1,i] = 0
    
    H1 = (3/T)*np.kron(np.eye(2*N), hopping_z1)
    
    hopping_z2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z2[i,i+1] = 0
        hopping_z2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_z2[i,i+1] = J2
        hopping_z2[i+1,i] = J2
    H2 = (3/T)*np.kron(np.eye(2*N), hopping_z2)
    
    hopping_z3 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z3[i,i] = 1
    for i in range(1,2*N,2):
        hopping_z3[i,i] = -1
    
    hopping_x = np.zeros((2*N,2*N),dtype=complex)
    for i in range(1,2*N-2,2):
        hopping_x[i,i+1] = 0.5*Jx+0.5*Jx
        hopping_x[i+1,i] = 0.5*Jx+0.5*Jx
    for i in range(0,2*N-1,2):
        hopping_x[i,i+1] = m-1j*Jy*np.sin(ky)+Jy*np.cos(ky)
        hopping_x[i+1,i] = m+1j*Jy*np.sin(ky)+Jy*np.cos(ky)
        
    H3 = (3/T)*np.kron(hopping_x, hopping_z3)

    UF = np.dot(expm(-1j*H1*(T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),expm(-1j*H1*(T/6))))))
    return UF

def main():
    # probs
    X, Z = np.meshgrid(np.linspace(0,2*N,2*N), np.linspace(0,2*N,2*N)) 
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x', fontsize=15, color='black',fontweight='bold')  
    ax.set_ylabel('z', fontsize=15, color='black',fontweight='bold')  
    ax.set_zlabel('probability', fontsize=15, color='black',fontweight='bold')

    K_array = np.array([-np.pi,np.pi])
    for ky in K_array:
        UF = Floquet(ky)
        eval,evec = np.linalg.eig(UF)
        eval,evec = ssort(eval,evec)
        
        # edgewave1 = evec[:,(eval>0/T-0.25)&((eval<0/T+0.25))]
        # for i in range(edgewave1.shape[1]):
        #     Plot0 = edgewave1[:,i]
        #     NumToPlot0 = np.abs(Plot0)**2
        #     p0 = np.reshape(NumToPlot0,(2*N,2*N)).T
        #     ax.plot_surface(X, Z, p0, cmap='plasma')
        #     ax.contour(X, Z, p0, 1000, zdir = 'z', offset = 0.25, cmap = plt.get_cmap('rainbow'))
        
        edgewave2 = evec[:,(eval>-np.pi/T-0.025)&((eval<-np.pi/T+0.025))]

        for i in range(edgewave2.shape[1]):
            Plot0 = edgewave2[:,i]
            NumToPlot0 = np.abs(Plot0)**2
            p0 = np.reshape(NumToPlot0,(2*N,2*N)).T
            ax.plot_surface(X, Z, p0, cmap='plasma')
            ax.contour(X, Z, p0, 1000, zdir = 'z', offset = 0.25, cmap = plt.get_cmap('rainbow'))
        
        edgewave3 = evec[:,(eval>np.pi/T-0.025)&((eval<np.pi/T+0.025))]

        for i in range(edgewave3.shape[1]):
            Plot0 = edgewave3[:,i]
            NumToPlot0 = np.abs(Plot0)**2
            p0 = np.reshape(NumToPlot0,(2*N,2*N)).T
            ax.plot_surface(X, Z, p0, cmap='plasma')
            ax.contour(X, Z, p0, 1000, zdir = 'z', offset = 0.25, cmap = plt.get_cmap('rainbow'))
    
    # bands
    # E = []
    # for ky in ky_array:
    #     UF = Floquet(ky)
    #     eval,evec = np.linalg.eig(UF)
    #     Eval = []
    #     for j in range(len(eval)):
    #         Eval.append((1j/T)*llog(eval[j],np.pi))
    #     Eval.sort()
    #     E.append(Eval)
    # plt.plot(ky_array,E,color='red')
    # plt.xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # plt.yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # plt.xlabel(r'$k_y$',fontsize=20,fontweight='bold')
    # plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    # plt.axhline(0, color='black')
    # plt.axhline(np.pi, color='black')
    # plt.axhline(-np.pi, color='black')
    plt.show()
    
if __name__ == '__main__':
    main()     
