import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
import seaborn as sns
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D 

#parameters
T = 1
J1 = 0.4*np.pi
J2 = 0.9*np.pi
phi = 0.2
m = -2
M = 20


d = 4

Nx = 30
Ny = 30
Nz = 30

kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
kz_array = np.linspace(-np.pi,np.pi,Nz+1)
# print(ky_array)
t_array = np.linspace(0.1,T-0.1,M+1)
gap = 0
lx = len(kx_array)
ly = len(ky_array)
lt = len(t_array)


#pauli matrix
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

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
    xx = [0 for index in range(len(x))]
    for i in range(len(x)):
        xx[i] = np.exp(1j*e[i])
    y = y[:,x_sort_idx]
    x = xx
    return x,y

def SSort(x,y):
    e = []
    for X in x:
        E = 1j*llog(X,np.pi)
        e.append(E)
    x_sort_idx = np.argsort(e)[::1]
    e = np.sort(e)[::1]
    y = y[:,x_sort_idx]
    x = e
    return x,y

#phase-fixing function
def fix(x,d):
    Phase = np.zeros(shape=(d,d),dtype=complex)
    for j in range(d):
        Phase[j,j] = np.exp(-1j*cmath.phase(x[0,j]))
    x = np.dot(x,Phase)
    return x

#anomalous periodic operator
def U(kx,ky,kz,t):
    #Hamiltonians
    H1 = (3/T)*J1*np.kron(s1,s0)
    H2 = (3/T)*J2*(np.cos(kz)*np.kron(s1,s0)+np.sin(kz)*np.kron(s2,s0))
    H3 = (3/T)*phi*(np.sin(kx)*np.kron(s3,s1)+np.sin(ky)*np.kron(s3,s2)+(m+np.cos(kx)+np.cos(ky))*np.kron(s3,s3))
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),expm(-1j*H1*(T/6))))))
    
    #sort by quasienergies
    eval,evec = np.linalg.eig(UF)
    evec = fix(evec,d)
    eval,evec = ssort(eval,evec)
    #fix phase

    #evolution operator
    r = int(t//T)

    if 0 <= t%T < T/6:
        U1 = np.dot(expm(-1j*H1*(t%T)),np.linalg.matrix_power(UF,r))
    if T/6 <= t%T < T/3:
        U1 = np.dot(expm(-1j*H2*(t%T-T/6)),np.dot(expm(-1j*H1*(T/6)),np.linalg.matrix_power(UF,r)))
    if T/3 <= t%T < 2*T/3:
        U1 = np.dot(expm(-1j*H3*(t%T-T/3)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H1*(T/6)),np.linalg.matrix_power(UF,r))))
    if 2*T/3 <= t%T < 5*T/6:
        U1 = np.dot(expm(-1j*H2*(t%T-2*T/3)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H1*(T/6)),np.linalg.matrix_power(UF,r)))))
    if 5*T/6 <= t%T < T:
        U1 = np.dot(expm(-1j*H1*(t%T-5*T/6)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H3*(T/3)),np.dot(expm(-1j*H2*(T/6)),np.dot(expm(-1j*H1*(T/6)),np.linalg.matrix_power(UF,r))))))
    
    #anomalous periodic operator for two gaps
    if gap == 0:
        Eaval = []
        for i in range(len(eval)):
            Eaval.append(np.exp(-llog(eval[i],0)*(t/T)))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(evec, np.dot(Smatpi, np.linalg.inv(evec)))
    if gap == np.pi:
        Eaval = []
        for i in range(len(eval)):
            Eaval.append(np.exp(-llog(eval[i],np.pi)*(t/T)))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(evec, np.dot(Smatpi, np.linalg.inv(evec)))
    return np.dot(U1,U2)

def main():
    X, Y = np.meshgrid(kx_array, ky_array)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    eigmatrix1 = np.zeros((lx,ly))
    eigmatrix2 = np.zeros((lx,ly))
    eigmatrix3 = np.zeros((lx,ly))
    eigmatrix4 = np.zeros((lx,ly))
    for i in range(lx):
        kx = kx_array[i]
        for j in range(ly):
            ky = ky_array[j]
            eigenvalue, eigenvector = np.linalg.eig(U(kx,ky,np.pi,0.5))
            quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
            # print(quasienergy)
            eigmatrix1[i,j] = (quasienergy[0]).real
            eigmatrix2[i,j] = (quasienergy[1]).real
            eigmatrix3[i,j] = (quasienergy[2]).real
            eigmatrix4[i,j] = (quasienergy[3]).real
        # eigmatrix = np.reshape(eigmatrix,(lx*ly,1))
    ax.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax.set_yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax.set_zticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'],size=18,weight='bold')
    ax.set_xlabel(r'$k_x$', fontsize=20,color='black',weight='bold')  
    ax.set_ylabel(r'$k_y$', fontsize=20,color='black',weight='bold') 
    ax.plot_surface(X, Y, eigmatrix1, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    ax.plot_surface(X, Y, eigmatrix4, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    
    plt.show() 

if __name__ == '__main__':
    main()