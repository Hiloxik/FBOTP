import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
import seaborn as sns
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.axisartist as axisartist

#parameters
T = 20
J11 = (2*np.pi/4)/(np.sqrt(2))
J12 = (2*np.pi/4)/(np.sqrt(2))
J21 = (3*np.pi/4)/(np.sqrt(2))
J22 = (3*np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
Nx = 30
Ny = 30
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(8,12,3)
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
def U(kx,ky,t,theta):
    #Hamiltonians
    H1 = (2/T)*(J11*np.kron(s1,s0)-J12*np.kron(s2,s2))+(0.0001)*(np.kron(s3,s0)+0.5*np.kron(s3,s3))
    H2 = (2/T)*(J21*(np.cos(theta)*np.cos(kx)+np.sin(theta)*np.cos(nx*kx))*np.kron(s1,s0)-J21*(np.cos(theta)*np.sin(kx)+np.sin(theta)*np.sin(nx*kx))*np.kron(s2,s3)
    -J22*(np.cos(theta)*np.cos(ky)+np.sin(theta)*np.cos(ny*ky))*np.kron(s2,s2)-J22*(np.cos(theta)*np.sin(ky)+np.sin(theta)*np.sin(ny*ky))*np.kron(s2,s1))
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
    fig=plt.figure(figsize=(5,5))
    # axs = Axes3D(fig)
    X, Y = np.meshgrid(kx_array, ky_array)
     
    ax1 = fig.add_subplot(131, projection='3d')
    for theta in theta_array:
        eigmatrix1 = np.zeros((lx,ly))
        eigmatrix2 = np.zeros((lx,ly))
        eigmatrix3 = np.zeros((lx,ly))
        eigmatrix4 = np.zeros((lx,ly))
        for i in range(lx):
            kx = kx_array[i]
            for j in range(ly):
                ky = ky_array[j]
                eigenvalue, eigenvector = np.linalg.eig(U(kx,ky,8,theta))
                quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
                # print(quasienergy)
                eigmatrix1[i,j] = (quasienergy[0]).real
                eigmatrix2[i,j] = (quasienergy[1]).real
                eigmatrix3[i,j] = (quasienergy[2]).real
                eigmatrix4[i,j] = (quasienergy[3]).real
        # eigmatrix = np.reshape(eigmatrix,(lx*ly,1))
    ax1.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax1.set_yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax1.set_zticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'],size=18,weight='bold')
    ax1.set_xlabel(r'$k_x$', fontsize=20,color='black',weight='bold')  
    ax1.set_ylabel(r'$k_y$', fontsize=20,color='black',weight='bold') 
    ax1.plot_surface(X, Y, eigmatrix1, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    ax1.plot_surface(X, Y, eigmatrix4, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    
    ax2 = fig.add_subplot(132, projection='3d')
    for theta in theta_array:
        eigmatrix1 = np.zeros((lx,ly))
        eigmatrix2 = np.zeros((lx,ly))
        eigmatrix3 = np.zeros((lx,ly))
        eigmatrix4 = np.zeros((lx,ly))
        for i in range(lx):
            kx = kx_array[i]
            for j in range(ly):
                ky = ky_array[j]
                eigenvalue, eigenvector = np.linalg.eig(U(kx,ky,10,theta))
                quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
                # print(quasienergy)
                eigmatrix1[i,j] = (quasienergy[0]).real
                eigmatrix2[i,j] = (quasienergy[1]).real
                eigmatrix3[i,j] = (quasienergy[2]).real
                eigmatrix4[i,j] = (quasienergy[3]).real
        # eigmatrix = np.reshape(eigmatrix,(lx*ly,1))
    ax2.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax2.set_yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax2.set_zticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'],size=18,weight='bold')
    ax2.set_xlabel(r'$k_x$', fontsize=20,color='black',weight='bold')  
    ax2.set_ylabel(r'$k_y$', fontsize=20,color='black',weight='bold') 
    ax2.plot_surface(X, Y, eigmatrix1, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    ax2.plot_surface(X, Y, eigmatrix4, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    
    ax3 = fig.add_subplot(133, projection='3d')
    for theta in theta_array:
        eigmatrix1 = np.zeros((lx,ly))
        eigmatrix2 = np.zeros((lx,ly))
        eigmatrix3 = np.zeros((lx,ly))
        eigmatrix4 = np.zeros((lx,ly))
        for i in range(lx):
            kx = kx_array[i]
            for j in range(ly):
                ky = ky_array[j]
                eigenvalue, eigenvector = np.linalg.eig(U(kx,ky,12,theta))
                quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
                # print(quasienergy)
                eigmatrix1[i,j] = (quasienergy[0]).real
                eigmatrix2[i,j] = (quasienergy[1]).real
                eigmatrix3[i,j] = (quasienergy[2]).real
                eigmatrix4[i,j] = (quasienergy[3]).real
        # eigmatrix = np.reshape(eigmatrix,(lx*ly,1))
    ax3.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax3.set_yticks([-np.pi,0,np.pi],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    ax3.set_zticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'],size=18,weight='bold')
    ax3.set_xlabel(r'$k_x$', fontsize=20,color='black',weight='bold')  
    ax3.set_ylabel(r'$k_y$', fontsize=20,color='black',weight='bold')  
    ax3.set_zlabel(r'$\pi$'+'-gap', fontsize=20,color='black',weight='bold')
    ax3.plot_surface(X, Y, eigmatrix1, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    ax3.plot_surface(X, Y, eigmatrix4, cmap=plt.get_cmap('plasma'), linewidth=0.1)  
    
    
    plt.show() 

if __name__ == '__main__':
    main()