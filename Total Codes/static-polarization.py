import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
from scipy.linalg import expm

T = 2
J1 = ((np.pi/2)/(np.sqrt(2)))*(2/T)
J2 = ((3*np.pi/4)/(np.sqrt(2)))*(2/T)
N = 5
d = 2*N*2
ky_array = np.linspace(-np.pi,np.pi,2*N+1)
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

def ssort(x,y):
    e = -1j*np.log(x)
    x_sort_idx = np.argsort(e)[::-1]
    e = np.sort(e)[::-1]
    xx = [0 for index in range(len(x))]
    for i in range(len(x)):
        xx[i] = np.exp(1j*e[i])
    y = y[:,x_sort_idx]
    x = xx
    return x,y

def fix(x,d):
    Phase = np.zeros(shape=(d,d),dtype=complex)
    for j in range(d):
        Phase[j,j] = np.exp(-1j*cmath.phase(x[0,j]))
    x = np.dot(x,Phase)
    return x

#calculate the evolution operator
def Floquet(ky):
    hopping_x1 = np.zeros((2*N,2*N))
    hopping_xy1 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_x1[i,i+1] = J1
        hopping_x1[i+1,i] = J1
    for i in range(1,2*N-1,2):
        hopping_x1[i,i+1] = 0
        hopping_x1[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_xy1[i, i] = -1
    for i in range(1,2*N,2):
        hopping_xy1[i, i] = 1
    h1 = np.kron(hopping_x1, s0)+np.kron(hopping_xy1, J1*s1)
    hopping_x2 = np.zeros((2*N,2*N))
    for i in range(0,2*N-1,2):
        hopping_x2[i,i+1] = 0
        hopping_x2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_x2[i,i+1] = J2
        hopping_x2[i+1,i] = J2
    h2 = np.kron(hopping_x2, s0)+np.kron(hopping_xy1, J2*(np.cos(ky)*s1+np.sin(ky)*s2))
    UF = np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4)))
    return UF
# print(np.round(Floquetian,2))

#solve the eigen-problem
def main():
    P = [0 for index in range(2*N)]
    for i in range(2*N):
        P[i] = 0
        for j in range(ly):
            ky = ky_array[j]
            eval0, evec0 = np.linalg.eig(Floquet(ky))
            evec0 = fix(evec0,d)
            eval0,evec0 = ssort(eval0,evec0)
            u = evec0[i]
            W = np.eye(2*N*2)
            for m in range(2*N):
                dk = m*((2*np.pi)/(2*N))
                eval1, evec1 = np.linalg.eig(Floquet(ky+dk))
                evec1 = fix(evec1,d)
                eval1,evec1 = ssort(eval1,evec1)
                eval2, evec2 = np.linalg.eig(Floquet(ky+dk+(2*np.pi)/(2*N)))
                evec2 = fix(evec2,d)
                eval2,evec2 = ssort(eval2,evec2)
                F = np.dot(np.conjugate(evec2).T,evec1)
                u,s,v = np.linalg.svd(F) #SVDa
                FF = np.dot(u,v)
                W = np.dot(FF,W)
            print(np.linalg.det(W),i,j)
            eigenvalue,eigenvector = np.linalg.eig(W)
            eigenvector = fix(eigenvector,d)
            EIG = (1/(2*np.pi*1j))*np.log(eigenvalue)
            Ematrix = np.diag(EIG)
            w = np.dot(u,eigenvector)
            p = np.dot(w,np.dot(Ematrix,np.conjugate(w).T))
            P[i] = p+P[i]
        P[i] = 1/(2*np.pi*2*N)*P[i][0]
    x_array = np.linspace(0,2*N,2*N)
    plt.scatter(x_array,P)
    # plt.title("OBC-"+str(N)+"cells-"+"a="+str(round(aalpha,2))+"-b="+str(round(bbeta,2)), fontdict={'size': 20})
    # plt.savefig('OBC-'+str(N)+"-"+str(round(aalpha,2))+"-"+str(round(bbeta,2))+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()