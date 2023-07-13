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
T = 20
M = 100
N = 50
J12 = (4*np.pi/4)/(np.sqrt(2))
J11_list = np.linspace(0,J12,M+1)
J2 = (1*np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
Nx = 30
Ny = 30
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(0.1,T-0.1,N+1)
gap = np.pi
lx = len(kx_array)
ly = len(ky_array)
lt = len(t_array)
theta_array = np.linspace(np.pi/2,np.pi/2,1)

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
        E = 1j*llog(X,0)
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
def U(J11,J12,kx,ky,t):
    #Hamiltonians
    H1 = (2/T)*(J11*np.kron(s1,s0)-J12*np.kron(s2,s2))+(0.0001)*(np.kron(s3,s0)+0.5*np.kron(s3,s3))
    H2 = (2/T)*(J2*(np.cos(nx*kx))*np.kron(s1,s0)-J2*(np.sin(nx*kx))*np.kron(s2,s3)
    -J2*(np.cos(ny*ky))*np.kron(s2,s2)-J2*(np.sin(ny*ky))*np.kron(s2,s1))
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/4)),np.dot(expm(-1j*H2*(T/2)),expm(-1j*H1*(T/4))))
    #sort by quasienergies
    eval,evec = np.linalg.eig(UF)
    evec = fix(evec,d)
    eval,evec = ssort(eval,evec)
    #fix phase

    #evolution operator
    r = int(t//T)

    if 0 <= t%T < T/4:
        U1 = np.dot(expm(-1j*H1*(t%T)),np.linalg.matrix_power(UF,r))
    if T/4 <= t%T < 3*T/4:
        U1 = np.dot(expm(-1j*H2*(t%T-T/4)),np.dot(expm(-1j*H1*(T/4)),np.linalg.matrix_power(UF,r)))
    if 3*T/4 <= t%T < T:
        U1 = np.dot(expm(-1j*H1*(t%T-3*T/4)),np.dot(expm(-1j*H2*(T/2)),np.dot(expm(-1j*H1*(T/4)),np.linalg.matrix_power(UF,r))))
    
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
    Gap = np.zeros((N+1,M+1))
    for i in range(len(t_array)):
        t = t_array[i]
        for j in range(len(J11_list)):
            J11 = J11_list[j] 
            eigenvalue, eigenvector = np.linalg.eig(U(J11,J12,np.pi,np.pi,t))
            quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
            Gap[i,j] = np.abs(quasienergy[2]-quasienergy[1])
    # np.savetxt('D:\科研\Topomat\Floquet insulator\FBOTP\\files1\\'+'DSgap-'+str(np.round(J2*(np.sqrt(2))/np.pi,2))+str(np.round(gap,1))+'.txt', np.c_[Gap],
    # fmt='%.18e',delimiter='\t')
    sns.heatmap(Gap,cmap='viridis')
    plt.show() 

if __name__ == '__main__':
    main()