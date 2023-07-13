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
J1 = 0.4*np.pi
J2 = 0.9*np.pi
phi = 0.2
m = 2
d = 4
nx = 1
ny = 1
nz = 1
Nx = 5
Ny = 5
Nz = 5
v = 1
w = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Ny+1)
kz_array = np.linspace(-np.pi,np.pi,Nz+1)
# print(ky_array)
t_array = np.linspace(0.001,T-0.001,21)

lx = len(kx_array)-1
ly = len(ky_array)-1
lz = len(kz_array)-1
lt = len(t_array)

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

    for gap in [0,np.pi]:
        na1_array = []
        na2_array = []
        for m in range(lt):
            start = time.time()
            t = t_array[m]
            Ux_array = []
            for h in range(lz):
                Ux = []
                for r in range(2*ly):
                    Uxx = []
                    for i in range(2*lx):
                        Uxx.append(U(-np.pi+i*((2*np.pi)/(Nx)),-np.pi+r*((2*np.pi)/(Ny)),-np.pi+h*((2*np.pi)/(Ny)),t,gap))
                    Uxx.append(Uxx[0])
                    Ux.append(Uxx)
                Ux.append(Ux[0])
                Ux_array.append(Ux)
            Ux_array.append(Ux_array[0])
            end = time.time()
            print(end-start)
            
            Evalz = np.array([0,0])
            for p in range(ly):
                branchy_array = []
                for i in range(lx):
                    branchx_array = []
                    for h in range(lz):
                        for r in range(ly):
                            Wx = np.kron(s0,s0)
                            for j in range(lx):
                                UU0x = Ux_array[h][r][i+j]
                                UU1x = np.conjugate(Ux_array[h][r][i+j+1]).T
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
                            if v == 1:
                                branchx = evecx[:,0:4]
                        branchx_array.append(branchx)
                    branchx_array.append(branchx_array[0])
                    for h in range(lz):
                        Wy = np.kron(s0,s0)
                        for e in range(ly):
                            UU0y = Ux_array[h][e][i]
                            UU1y = np.conjugate(Ux_array[h][e+1][i]).T
                            qy = (1-1/(2*(ny)))*np.kron(s0,s0)+(1/(2*(ny)))*np.dot(UU1y,UU0y)
                            Qy = np.dot(np.conjugate(branchx_array[e+1]).T,np.dot(qy,branchx_array[e]))
                            # print(np.dot(np.conjugate(branchx_array[j]).T,branchx_array[j]))
                            uy,sy,vy = np.linalg.svd(Qy) #SVDa
                            QQy = np.dot(uy,vy)
                            # print(np.linalg.det(QQy))
                            Wy = np.dot(QQy,Wy)
                        evaly, evecy = np.linalg.eig(Wy)
                        if w == 1:
                            branchy = evecy[:,0:2]
                        if w == 2:
                            branchy = evecy[:,2:4]
                    branchy_array.append(branchy)
                branchy_array.append(branchy_array[0])
                Wz = s0
                for h in range(lz):
                    for q in range(lx):
                        UU0z = Ux_array[h][p][q]
                        UU1z = np.conjugate(Ux_array[h+1][p][q]).T
                        qz = (1-1/(2*(nz)))*np.kron(s0,s0)+(1/(2*(nz)))*np.dot(UU1z,UU0z)
                        Qz = np.dot(np.conjugate(branchy_array[p+1]).T,np.dot(qz,branchy_array[p]))
                        # print(np.dot(np.conjugate(branchx_array[j]).T,branchx_array[j]))
                        uz,sz,vz = np.linalg.svd(Qz) #SVDa
                        QQz = np.dot(uz,vz)
                        # print(np.linalg.det(QQy))
                        Wz = np.dot(QQz,Wz)
                print("det=",np.round(np.linalg.det(Wz),7),np.round(100*(((p+1+m*lz)/(lz*lt))),2),"%")
                evalz, evecz = np.linalg.eig(Wz)            
                                
                ea = []
                for i in range(len(evalz)):
                    ea.append(((1j/(2*np.pi))*llog(evalz[i],0)).real)
                ea.sort()
                # print(ea)
                Evalz = ea+Evalz
            na1_array.append((1/(Nz))*Evalz[0])
            na2_array.append((1/(Nz))*Evalz[1])
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
    plt.xticks([0,T/2,T], [r'$0$' ,r'$T/2$',r'$T$'],weight='bold')
    plt.yticks([0,0.5,1],weight='bold')
    plt.xlabel("time", fontdict={'size': 20})
    plt.ylabel("polarization", fontdict={'size':20})
    # plt.title("dynamical polarization", fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()