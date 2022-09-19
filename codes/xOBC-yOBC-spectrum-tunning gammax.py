from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn as sns
import scipy
import functools
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D 

T = 2
J12 = (1*np.pi/4)/(np.sqrt(2))
J11_list = np.linspace(0,4*J12,20)
J2 = (4*np.pi/4)/(np.sqrt(2))
J3 = 0
N = 10
d = 2*N*2*N

cm = plt.cm.get_cmap('plasma') #get colorbar

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
    
def fix(x,d):
    Phase = np.zeros(shape=(d,d),dtype=complex)
    for j in range(d):
        Phase[j,j] = np.exp(-1j*cmath.phase(x[0,j]))
    x = np.dot(x,Phase)
    return x

#calculate the evolution operator
def Floquet(J11):
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
    h2 = (2/T)*(np.kron(hopping_x2, np.eye(2*N))+np.kron(hopping_xy, hopping_y2))
    UF = np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4)))
    return UF

def main():
    E = []
    for m in range(len(J11_list)):
        J11 = J11_list[m]
        UF = Floquet(J11)    
        eval,evec = np.linalg.eig(UF)
        Eval = []
        for k in range(len(eval)):
            Eval.append((1j)*llog(eval[k],0))
        Eval.sort()
        E.append(Eval)
    plt.plot(J11_list,E,color='blue')
    plt.xticks(size=18,weight='bold')
    plt.yticks(size=18,weight='bold')
    plt.xlabel(r'$\gamma_x$',fontsize=20,fontweight='bold')
    plt.ylabel(r'$E$',fontsize=20,fontweight='bold')
    plt.axvline(J12, color='black', linestyle='--')
    plt.show()

if __name__ == '__main__':
    main()