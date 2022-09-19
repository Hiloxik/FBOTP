import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn as sns
import scipy
import functools
from scipy.linalg import expm

T = 1
M = 30
J11_list = np.linspace(-(4*np.pi/4)/(np.sqrt(2)),(4*np.pi/4)/(np.sqrt(2)),M+1)
J12_list = np.linspace((4*np.pi/4)/(np.sqrt(2)),-(4*np.pi/4)/(np.sqrt(2)),M+1)
J2 = (8*np.pi/4)/(np.sqrt(2))
J3 = 0
N = 5
n = 1
ky_array = np.linspace(-np.pi,np.pi,2*10+1)
ly = len(ky_array)

cm = plt.cm.get_cmap('plasma') #get colorbar

s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j*argument

#calculate the evolution operator
def Floquet(J11,J12):
    hopping_x1 = np.zeros((2*N,2*N))
    hopping_y1 = np.zeros((2*N,2*N))
    hopping_x2 = np.zeros((2*N,2*N))
    hopping_y2 = np.zeros((2*N,2*N))
    hopping_xy = np.zeros((2*N,2*N))

    for i in range(0,2*N-1,2):
        hopping_x1[i,i+1] = J11
        hopping_x1[i+1,i] = J11
    for i in range(1,2*N-1,2):
        hopping_x1[i,i+1] = 0
        hopping_x1[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_y1[i,i+1] = J12
        hopping_y1[i+1,i] = J12
    for i in range(1,2*N-1,2):
        hopping_y2[i,i+1] = 0
        hopping_y2[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_xy[i, i] = -1
    for i in range(1,2*N,2):
        hopping_xy[i, i] = 1
    h1 = (2/T)*(np.kron(hopping_x1, np.eye(2*N))+np.kron(hopping_xy, hopping_y1))

    for i in range(0,2*N-1,2):
        hopping_x2[i,i+1] = 0
        hopping_x2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_x2[i,i+1] = J2
        hopping_x2[i+1,i] = J2
    for i in range(0,2*N-1,2):
        hopping_y2[i,i+1] = 0
        hopping_y2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_y2[i,i+1] = J2
        hopping_y2[i+1,i] = J2
    h2 = np.kron(hopping_x2, np.eye(2*N))+np.kron(hopping_xy, hopping_y2)
    UF = (2/T)*(np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4))))
    return UF
# print(np.round(Floquetian,2))

#solve the eigen-problem
def main():
    gap = np.zeros((M,M))
    for i in range(M):
        J12 = J12_list[i]
        for j in range(M):
            J11 = J11_list[j]
        
            UF = Floquet(J11,J12)
            eval,evec = np.linalg.eig(UF)
            Eval = []
            for k in range(len(eval)):
                Eval.append((1j)*llog(eval[k],0)-np.pi)
            Eval.sort()
            E = np.abs(Eval)
            gap[i,j] = 2*np.min(E)
    
    sns.heatmap(gap,cmap='afmhot')
    plt.xlabel(r'$\gamma_x$')
    plt.ylabel(r'$\gamma_y$')
    plt.show()

if __name__ == '__main__':
    main()