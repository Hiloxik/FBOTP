import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
from qutip import *
import pandas as pd
import time
import seaborn as sns

#parameters
T = 20
M = 50
J11_list = np.linspace(-(8*np.pi/4)/(np.sqrt(2)),(8*np.pi/4)/(np.sqrt(2)),M+1)
J12_list = np.linspace((8*np.pi/4)/(np.sqrt(2)),-(8*np.pi/4)/(np.sqrt(2)),M+1)
xlist = np.linspace(-8,8,M)
ylist = np.linspace(8,-8,M)
J21 = (3.9*np.pi/4)/(np.sqrt(2))
J22 = (3.9*np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
Nx = 5
Ny = 5
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(T/2,T/2,1)
gap = 0
lx = len(kx_array)-1
ly = len(ky_array)-1
lt = len(t_array)
theta_array = np.linspace(np.pi/2,np.pi/2,1)

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

#anomalous periodic operator
def U(J11,J12,kx,ky,t,theta):
    #Hamiltonians
    H1 = (2/T)*(J11*np.kron(s1,s0)-J12*np.kron(s2,s2))+(0.0001)*(np.kron(s0,s3)+0.5*np.kron(s3,s3))
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

#polarization calculation
def main():
    gap = np.zeros((M,M))
    for q in range(M):
        J12 = J12_list[q]
        for s in range(M):
            J11 = J11_list[s]
            for theta in theta_array:
                na1_array = []
                na2_array = []
                for m in range(lt):
                    start = time.time()
                    t = t_array[m]
                    Uy_array = []
                    for r in range(lx):
                        Uy = []
                        for i in range(2*ly):
                            Uy.append(U(J11,J12,-np.pi+r*((2*np.pi)/(Nx)),-np.pi+i*((2*np.pi)/(Ny)),t,theta))
                        Uy.append(Uy[0])
                        Uy_array.append(Uy)
                    Uy_array.append(Uy_array[0])
                    end = time.time()
                    print(end-start)
                    Evalx = np.array([0,0])
                    for i in range(ly):
                        branchy_array = []
                        for r in range(lx):
                            Wy = np.kron(s0,s0)
                            for j in range(ly):
                                UU0y = Uy_array[r][i+j]
                                UU1y = np.conjugate(Uy_array[r][i+j+1]).T
                                Qy = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1y,UU0y)
                                # print(np.linalg.det(Qx)) 
                                uy,sy,vy = np.linalg.svd(Qy) #SVD
                                QQy = np.dot(uy,vy)
                                Wy = np.dot(QQy,Wy)
                            # print(np.linalg.det(Wx)) 
                            evaly, evecy = np.linalg.eig(Wy)
                            evecy = fix(evecy,d)
                            # print(evecx)
                            evaly, evecy = SSort(evaly,evecy)
                            if v == 1:
                                branchy = evecy[:,0:2]
                            if v == 2:
                                branchy = evecy[:,2:4]
                            branchy_array.append(branchy)
                        branchy_array.append(branchy_array[0])
                        Wx = s0
                        for e in range(lx):
                            UU0x = Uy_array[e][i]
                            UU1x = np.conjugate(Uy_array[e+1][i]).T
                            qx = (1-1/(2*(nx)))*np.kron(s0,s0)+(1/(2*(nx)))*np.dot(UU1x,UU0x)
                            Qx = np.dot(np.conjugate(branchy_array[e+1]).T,np.dot(qx,branchy_array[e]))
                            # print(np.dot(np.conjugate(branchx_array[j]).T,branchx_array[j]))
                            ux,sx,vx = np.linalg.svd(Qx) #SVDa
                            QQx = np.dot(ux,vx)
                            # print(np.linalg.det(QQy))
                            Wx = np.dot(QQx,Wx)
                        print("det=",np.round(np.linalg.det(Wx),7),np.round(100*((s+1+q*M)/(M**2)),2),"%")
                        evalx, evecx = np.linalg.eig(Wx)
                        ea = []
                        for i in range(len(evalx)):
                            ea.append(((1j/(2*np.pi))*llog(evalx[i],0)).real)
                        ea.sort()
                        # print(ea)
                        Evalx = ea+Evalx
                    na1_array.append((1/(Ny))*Evalx[0])
                    na2_array.append((1/(Ny))*Evalx[1])
            gap[q,s]=np.round(np.abs(na1_array[0]-na2_array[0]))
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    np.savetxt('D:\科研\Topomat\Floquet insulator\FBOTP\\files\\'+'DP'+str(np.round(J21*(np.sqrt(2))/np.pi,2))+'-yx-l.txt', np.c_[gap],
    fmt='%.18e',delimiter='\t')
    sns.heatmap(gap,cmap='viridis')
    plt.xlabel(r'$\frac{\pi}{\sqrt{2}}\gamma_x$')
    plt.ylabel(r'$\frac{\pi}{\sqrt{2}}\gamma_y$')
    plt.xticks(np.arange(len(xlist))+0.5, np.around(xlist,1),fontsize=5,rotation=45)
    plt.yticks(np.arange(len(ylist))+0.5, np.around(ylist,1),fontsize=5,rotation=45)
    plt.show()

if __name__ == '__main__':
    main()