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
M = 50
J12_list = np.linspace((11*np.pi/4)/(np.sqrt(2)),0,M+1)
J11_list = np.linspace(0,(11*np.pi/4)/(np.sqrt(2)),M+1)
xlist = np.linspace(0,11,M)
ylist = np.linspace(11,0,M)
J2 = (np.pi/4)/(np.sqrt(2))
J3 = 0
N = 20
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
def Floquet(J11, J12, ky):
    hopping_y1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_y1[i,i+1] = J12
        hopping_y1[i+1,i] = J12
    for i in range(1,2*N-1,2):
        hopping_y1[i,i+1] = 0
        hopping_y1[i+1,i] = 0
    h1 = (2/T)*(np.kron(s3, hopping_y1)+np.kron(J11*s1, np.eye(2*N)))
    hopping_y2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_y2[i,i+1] = 0
        hopping_y2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_y2[i,i+1] = J2
        hopping_y2[i+1,i] = J2
    h2 = (2/T)*(np.kron(s3, hopping_y2,)+np.kron(J2*(np.cos(n*ky)*s1+np.sin(n*ky)*s2), np.eye(2*N)))
    F = np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4)))
    return F
# print(np.round(Floquetian,2))

#solve the eigen-problem
def main():
    gap = np.zeros((M,M))
    for i in range(M):
        J12 = J12_list[i]
        for j in range(M):
            J11 = J11_list[j]
            UF1 = Floquet(J11,J12,0)
            UF2 = Floquet(J11,J12,np.pi)
            eval1,evec1 = np.linalg.eig(UF1)
            eval2,evec2 = np.linalg.eig(UF2)
            Eval = []
            for k in range(len(eval1)):
                Eval.append((1j)*llog(eval1[k],np.pi))
                Eval.append((1j)*llog(eval2[k],np.pi))
            Eval.sort()
            E = np.abs(Eval)
            gap[i,j] = 2*np.min(E)
    
    sns.heatmap(gap,cmap='afmhot')
    plt.xlabel(r'$\gamma_x$')
    plt.ylabel(r'$\gamma_y$')
    plt.xticks(np.arange(len(xlist))+0.5, np.around(xlist,1),fontsize=5,rotation=45)
    plt.yticks(np.arange(len(ylist))+0.5, np.around(ylist,1),fontsize=5,rotation=45)
    plt.show()

if __name__ == '__main__':
    main()