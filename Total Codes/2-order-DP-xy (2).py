import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm
from qutip import *
import pandas as pd
import time

#parameters
T = 20
J11 = (12*np.pi/4)/(np.sqrt(2))
J12 = (13*np.pi/4)/(np.sqrt(2))
J21 = (3*np.pi/4)/(np.sqrt(2))
J22 = (3*np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
Nx = 10
Ny = 10
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
# print(ky_array)
t_array = np.linspace(0.001,T-0.001,11)

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
def U(kx,ky,t,theta,gap):
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

#polarization calculation
def main():
    for gap in [0,np.pi]:
        
        for theta in theta_array:
            na1_array = []
            na2_array = []
            for m in range(lt):
                start = time.time()
                t = t_array[m]
                Ux_array = []
                for r in range(ly):
                    Ux = []
                    for i in range(2*lx):
                        Ux.append(U(-np.pi+i*((2*np.pi)/(Nx)),-np.pi+r*((2*np.pi)/(Ny)),t,theta,gap))
                    Ux.append(Ux[0])
                    Ux_array.append(Ux)
                Ux_array.append(Ux_array[0])
                end = time.time()
                print(end-start)
                
                Evaly = np.array([0,0])
                for i in range(lx):
                    branchx_array = []
                    for r in range(ly):
                        Wx = np.kron(s0,s0)
                        for j in range(lx):
                            UU0x = Ux_array[r][i+j]
                            UU1x = np.conjugate(Ux_array[r][i+j+1]).T
                            Qx = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1x,UU0x)
                            # print(np.linalg.det(Qx)) 
                            ux,sx,vx = np.linalg.svd(Qx) #SVD
                            QQx = np.dot(ux,vx)
                            Wx = np.dot(QQx,Wx)
                        # print(np.linalg.det(Wx)) 
                        evalx, evecx = np.linalg.eig(Wx)
                        evecx = fix(evecx,d)
                        # print(evecx)
                        evalx, evecx = SSort(evalx,evecx)
                        # print(evalx,evecx)
                        if v == 1:
                            branchx = evecx[:,0:2]
                        if v == 2:
                            branchx = evecx[:,2:4]
                        branchx_array.append(branchx)
                    branchx_array.append(branchx_array[0])
                    Wy = s0
                    for e in range(ly):
                        UU0y = Ux_array[e][i]
                        UU1y = np.conjugate(Ux_array[e+1][i]).T
                        qy = (1-1/(2*(ny)))*np.kron(s0,s0)+(1/(2*(ny)))*np.dot(UU1y,UU0y)
                        Qy = np.dot(np.conjugate(branchx_array[e+1]).T,np.dot(qy,branchx_array[e]))
                        # print(np.dot(np.conjugate(branchx_array[j]).T,branchx_array[j]))
                        uy,sy,vy = np.linalg.svd(Qy) #SVDa
                        QQy = np.dot(uy,vy)
                        # print(np.linalg.det(QQy))
                        Wy = np.dot(QQy,Wy)
                    print("det=",np.round(np.linalg.det(Wy),7),np.round(100*(((i+1+m*lx)/(lx*lt))),2),"%")
                    evaly, evecy = np.linalg.eig(Wy)
                    ea = []
                    for i in range(len(evaly)):
                        ea.append(((1j/(2*np.pi))*llog(evaly[i],0)).real)
                    ea.sort()
                    # print(ea)
                    Evaly = ea+Evaly
                na1_array.append((1/(Nx))*Evaly[0])
                na2_array.append((1/(Nx))*Evaly[1])
            if gap == 0:
                S1 = plt.scatter(t_array,na1_array)
                S2 = plt.scatter(t_array,na2_array)
            if gap == np.pi:
                S3 = plt.scatter(t_array,na1_array,marker='x',c='red',s=30)
                S4 = plt.scatter(t_array,na2_array,marker='x',c='green',s=30)
    plt.legend((S1,S2,S3,S4),(str(0)+'+',str(0)+'-',r'$\pi$' +'+',r'$\pi$'+'-'))
    tt = np.linspace(0,T,100)
    z1 = 0*tt+0.5
    z2 = 0*tt+1
    z3 = 0*tt+0
    plt.plot(tt,z1,c='black')
    plt.plot(tt,z2,c='black')
    plt.plot(tt,z3,c='black')
    plt.xticks([0,T/2,T], [r'$0$' ,r'$T/2$',r'$T$'],size=18,weight='bold')
    plt.yticks([0,0.5,1],size=18,weight='bold')
    plt.xlabel("time", fontdict={'size': 25},fontweight='bold')
    plt.ylabel("polarization", fontdict={'size':25},fontweight='bold')
    # plt.title("dynamical polarization", fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()