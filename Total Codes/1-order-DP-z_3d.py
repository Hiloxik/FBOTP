import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D

#parameters
T = 20
J1 = 0.4*np.pi
J2 = 0.9*np.pi
phi = 0.2
m = -np.pi/phi
d = 4
nx = 1
ny = 1
nz = 1
Nx = 10
Ny = 10
Nz = 10
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
kz_array = np.linspace(-np.pi,np.pi,Nz+1)
# print(ky_array)
t_array = np.linspace(0.001,T-0.001,21)
c_array = np.linspace(0,10,9)

lx = len(kx_array)-1
ly = len(ky_array)-1
lz = len(kz_array)-1
lt = len(t_array)
lc = len(c_array)

#pauli matrix
s0 = np.eye(2)
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

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

#anomalous periodic operator
def U(kx,ky,kz,t,gap):
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

#polarization calculation
def main():
    X, Y = np.meshgrid(kx_array,ky_array)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for gap in [np.pi]:
        na1_array = np.zeros((len(ky_array),len(kx_array)),dtype=float)
        na2_array = np.zeros((len(ky_array),len(kx_array)),dtype=float)
        na3_array = np.zeros((len(ky_array),len(kx_array)),dtype=float)
        na4_array = np.zeros((len(ky_array),len(kx_array)),dtype=float)
        for i in range(len(ky_array)):
            ky = ky_array[i]
            for j in range(len(kx_array)):
                kx = kx_array[j]
                Wz = np.kron(s0,s0)
                kz = -np.pi
                for k in range(lz):
                    dk = k*((2*np.pi)/(Nz))
                    UU0z = U(kx,ky,kz+dk,T/2,gap)
                    UU1z = np.conjugate(U(kx,ky,kz+dk+(2*np.pi)/(Nz),T/2,gap)).T
                    Qz = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1z,UU0z)
                    # print(np.linalg.det(Qz)) 
                    uz,sz,vz = np.linalg.svd(Qz) #SVD
                    QQz = np.dot(uz,vz)
                    Wz = np.dot(QQz,Wz)
                # print(np.linalg.det(Wz)) 
                evalz, evecz = np.linalg.eig(Wz)
                ee = [0 for index in range(len(evalz))]
                for l in range(len(evalz)):
                    ee[l] = (1j/(2*np.pi))*llog(evalz[l],0)
                ee.sort()
                    
                na1_array[i,j] = ee[0].real
                na2_array[i,j] = ee[1].real
                na3_array[i,j] = ee[2].real
                na4_array[i,j] = ee[3].real

        if gap == 0:
            ax.plot_surface(X, Y, na1_array, linewidth=0.1, cmap='viridis')
            ax.plot_surface(X, Y, na2_array, linewidth=0.1, cmap='viridis')
            ax.plot_surface(X, Y, na3_array, linewidth=0.1, cmap='viridis')
            ax.plot_surface(X, Y, na4_array, linewidth=0.1, cmap='viridis')
        if gap == np.pi:
            ax.plot_surface(X, Y, na1_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
            ax.plot_surface(X, Y, na2_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
            ax.plot_surface(X, Y, na3_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
            ax.plot_surface(X, Y, na4_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
            
    ax.set_xlabel('kx', fontsize=15, color='black')  
    ax.set_ylabel('ky', fontsize=15, color='black')  
    ax.set_zlabel(r'$\nu_z$', fontsize=15, color='black')  
    plt.title("dynamical polarization", fontdict={'size': 20})
    # np.savetxt('D:\科研\Topomat\Floquet insulator\\3D-FBOTP\\files\\'+'DP-1-'+str(np.round(m,1))+str(np.round(gap,1))+'-z-1.txt', np.c_[na1_array],fmt='%.18e',delimiter='\t')
    # np.savetxt('D:\科研\Topomat\Floquet insulator\\3D-FBOTP\\files\\'+'DP-1-'+str(np.round(m,1))+str(np.round(gap,1))+'-z-2.txt', np.c_[na3_array],fmt='%.18e',delimiter='\t')
    plt.show()

if __name__ == '__main__':
    main()