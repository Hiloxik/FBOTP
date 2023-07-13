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

J12 = (4*np.pi/4)/(np.sqrt(2))
J11_list = np.linspace(0,J12,20)
J2 = (1*np.pi/4)/(np.sqrt(2))
J3 = 0
N = 20
n = 1
ky_array = np.linspace(-np.pi,np.pi,2*N+1)
ly = len(ky_array)

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

#calculate the evolution operator
def Floquet(J11, ky):
    hopping_x1 = np.zeros((2*N,2*N))
    hopping_xy1 = np.zeros((2*N,2*N))
    hopping_xy2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_x1[i,i+1] = J11
        hopping_x1[i+1,i] = J11
    for i in range(1,2*N-1,2):
        hopping_x1[i,i+1] = 0
        hopping_x1[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_xy1[i, i] = -1
    for i in range(1,2*N,2):
        hopping_xy1[i, i] = 1
    h1 = (2/T)*(np.kron(hopping_x1, s0)+np.kron(hopping_xy1, J12*s1))
    hopping_x2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_x2[i,i+1] = 0
        hopping_x2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_x2[i,i+1] = J2
        hopping_x2[i+1,i] = J2
    for i in range(0,2*N-1,2):
        hopping_xy2[i, i] = -1
    for i in range(1,2*N,2):
        hopping_xy2[i, i] = 1
    h2 = (2/T)*(np.kron(hopping_x2, s0)+np.kron(hopping_xy2, J2*(np.cos(n*ky)*s1+np.sin(n*ky)*s2)))
    F = np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4)))
    return F


def main():
    
    E = []
    for ky in ky_array:
        J11 = 1.2
        UF = Floquet(J11,ky)
    
        eval,evec = np.linalg.eig(UF)
        Eval = []
        for j in range(len(eval)):
            Eval.append((1j/T)*llog(eval[j],0))
        Eval.sort()
        E.append(Eval)
    plt.plot(ky_array,E,color='red')
    plt.xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    plt.yticks([1.5,np.pi,5],['1.5',r'$\pi$','5'],size=18,weight='bold')
    plt.xlabel(r'$k_y$',fontsize=20,fontweight='bold')
    plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    plt.axvline(0, color='black', linestyle='--')
    plt.axhline(np.pi, color='black', linestyle='--')
    plt.show()
    
if __name__ == '__main__':
    main()     