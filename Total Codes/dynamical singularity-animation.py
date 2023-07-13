import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
import seaborn as sns
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation

#parameters
T = 20
J1 = (np.pi/2)/(np.sqrt(2))
J2 = (np.pi/4)/(np.sqrt(2))
d = 4
nx = 2
ny = 2
Nx = 20
Ny = 20
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(0.001*T,0.999*T,11)
gap = 0
lx = len(kx_array)-1
ly = len(ky_array)-1
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
        if gap == 0:
            E = -1j*llog(X,0)/T
        if gap == np.pi:
            E = -1j*llog(X,np.pi)/T
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
    H1 = (2*J1/T)*(np.kron(s1,s0)-np.kron(s2,s2))+(0*2/T)*(np.sin(ny*ky)*np.kron(s3,s1)+np.cos(ny*ky)*np.kron(s3,s2)-0.4*np.cos(ny*ky)*np.kron(s0,s0))+(0.0001)*np.kron(s3,s0)
    H2 = (2*J2/T)*((np.cos(theta)*np.cos(kx)+np.sin(theta)*np.cos(nx*kx))*np.kron(s1,s0)-(np.cos(theta)*np.sin(kx)+np.sin(theta)*np.sin(nx*kx))*np.kron(s2,s3)-(np.cos(theta)*np.cos(ky)+np.sin(theta)*np.cos(ny*ky))*np.kron(s2,s2)-(np.cos(theta)*np.sin(ky)+np.sin(theta)*np.sin(ny*ky))*np.kron(s2,s1))+(0.0001)*np.kron(s3,s0)
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
        Eaval = [0 for index in range(len(eval))]
        for i in range(len(eval)):
            Eaval[i] = np.exp(-llog(eval[i],0)*(t/T))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(evec, np.dot(Smatpi, np.linalg.inv(evec)))
    if gap == np.pi:
        Eaval = [0 for index in range(len(eval))]
        for i in range(len(eval)):
            Eaval[i] = np.exp(-llog(eval[i],np.pi)*(t/T))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(evec, np.dot(Smatpi, np.linalg.inv(evec)))
    return np.dot(U1,U2)

for theta in theta_array:
    Kx_array = kx_array.tolist()
    Ky_array = ky_array.tolist()
    del(Kx_array[-1])
    del(Ky_array[-1])
    X, Y = np.meshgrid(Kx_array, Ky_array) 
    EIG1 = []
    EIG2 = []
    for t in t_array:
        eigmatrix1 = np.zeros((lx,ly))
        eigmatrix2 = np.zeros((lx,ly))
        for i in range(lx):
            for j in range(ly):
                kx = kx_array[i]
                ky = ky_array[j]
                eigenvalue, eigenvector = np.linalg.eig(U(kx,ky,t,theta))
                quasienergy, eigenvector = SSort(eigenvalue,eigenvector)
                eigmatrix1[i,j] = (quasienergy[0]).real
                eigmatrix2[i,j] = (quasienergy[3]).real
        EIG1.append(eigmatrix1)
        EIG2.append(eigmatrix2)
    
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 6), subplot_kw=dict(projection='3d'))
    plot1 = [ax.plot_surface(X, Y, EIG1[0], cmap=plt.get_cmap('plasma'), linewidth=0)]
    plot2 = [ax.plot_surface(X, Y, EIG2[0], cmap=plt.get_cmap('plasma'), linewidth=0)]
    ax.set_xlabel('kx', fontsize=10, color='black')  
    ax.set_ylabel('ky', fontsize=10, color='black')  
    ax.set_zlabel('quasienergy', fontsize=10, color='black')


    ax.view_init(50, 120)   

    def update_map1(num, z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, EIG1[num], cmap=plt.get_cmap('plasma'), linewidth=0)
    
    def update_map2(num, z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, EIG2[num], cmap=plt.get_cmap('plasma'), linewidth=0)

    ani1 = animation.FuncAnimation(fig, update_map1, 30, interval=500, fargs=(EIG1, plot1), repeat=True)
    ani2 = animation.FuncAnimation(fig, update_map2, 30, interval=500, fargs=(EIG2, plot2), repeat=True)
    plt.show()