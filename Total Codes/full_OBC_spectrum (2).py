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
m = 0.342


d = 4

Nx = 10
Ny = 10
Nz = 10

N = 8

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
def Floquet():
    hopping_z1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z1[i,i+1] = J1
        hopping_z1[i+1,i] = J1
    for i in range(1,2*N-1,2):
        hopping_z1[i,i+1] = 0
        hopping_z1[i+1,i] = 0
    
    H1 = (3/T)*np.kron(np.eye(2*N),np.kron(np.eye(2*N), hopping_z1))
    
    hopping_z2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_z2[i,i+1] = 0
        hopping_z2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_z2[i,i+1] = J2
        hopping_z2[i+1,i] = J2
    H2 = (3/T)*np.kron(np.eye(2*N),np.kron(np.eye(2*N), hopping_z2))
    
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
    # states
    # fig = plt.figure(figsize=(7,7))
    # UF = Floquet()
    # eval,evec = np.linalg.eig(UF)
    # Eval = []
    # for j in range(len(eval)):
    #     Eval.append((1j/T)*llog(eval[j],np.pi))
    # Eval.sort()
    # plt.scatter(np.linspace(0,len(Eval),len(Eval)),Eval)
    # plt.xticks(size=18,weight='bold')
    # plt.yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # plt.xlabel(r'$states$',fontsize=20,fontweight='bold')
    # plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    # plt.axhline(0, color='black')
    # plt.axhline(np.pi, color='black')
    # plt.axhline(-np.pi, color='black')
    
    # prob
    X, Y, Z = np.meshgrid(np.linspace(0,2*N,2*N), np.linspace(0,2*N,2*N), np.linspace(0,2*N,2*N)) 
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x', fontsize=10, color='black')  
    ax.set_ylabel('y', fontsize=10, color='black')  
    ax.set_zlabel('z', fontsize=10, color='black')
    UF = Floquet()
    eval,evec = np.linalg.eig(UF)
    eval,evec = ssort(eval,evec)
    
    edgewave1 = evec[:,(eval>-np.pi/T-0.25)&((eval<-np.pi/T+0.25))]
    edgewave2 = evec[:,(eval>np.pi/T-0.25)&((eval<np.pi/T+0.25))]
    NUM = np.zeros((2*N,2*N,2*N))
    for i in range(edgewave1.shape[1]):
        Plot1 = edgewave1[:,i]
        NumToPlot1 = np.abs(Plot1)**2
        NumToPlot1 = np.reshape(NumToPlot1,(2*N,2*N,2*N))
        Plot2 = edgewave2[:,i]
        NumToPlot2 = np.abs(Plot2)**2
        NumToPlot2 = np.reshape(NumToPlot2,(2*N,2*N,2*N))
        NUM += NumToPlot1 + NumToPlot2
    
    # edgewave0 = evec[:,(eval>0/T-0.25)&((eval<0/T+0.25))]
    # NUM = np.zeros((2*N,2*N,2*N))
    # for i in range(edgewave0.shape[1]):
    #     Plot0 = edgewave0[:,i]
    #     NumToPlot0 = np.abs(Plot0)**2
    #     NumToPlot0 = np.reshape(NumToPlot0,(2*N,2*N,2*N))
    #     NUM += NumToPlot0
            
    ax.scatter(Y, X, Z, c=NUM, marker='s', cmap='Greys', alpha=1,s=100)

    plt.show()
    
if __name__ == '__main__':
    main()     
