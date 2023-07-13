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
m = np.pi/phi+1
m_array = np.linspace(0,50,5)
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
t_array = np.linspace(0.001,T-0.001,1)
c_array = np.linspace(0,10,11)

lx = len(kx_array)-1
ly = len(ky_array)-1
lz = len(kz_array)-1
lt = len(t_array)
lm = len(m_array)

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
    H1 = (3/T)*J1*np.kron(s1,s0)+(10**(-5))*(0.01*np.kron(s3,s0)+1000*np.kron(s3,s3))
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
    # X, Y = np.meshgrid(t_array,ky_array)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    for gap in [0,np.pi]:
        # na1_array = np.zeros((len(ky_array),len(t_array)))
        # na2_array = np.zeros((len(ky_array),len(t_array)))
        na1_array = []
        na2_array = []
        for p in range(lt):
            t = t_array[p]
            Pol1 = []
            Pol2 = []
            for q in range(ly):
                ky = ky_array[q]
                start = time.time()
                Uz_array = []
                for r in range(lx):
                    Uz = []
                    for i in range(2*lz):
                        Uz.append(U(-np.pi+i*((2*np.pi)/(Nx)),ky,-np.pi+r*((2*np.pi)/(Nz)),t,gap))
                    Uz.append(Uz[0])
                    Uz_array.append(Uz)
                Uz_array.append(Uz_array[0])
                end = time.time()
                print(end-start)
                
                Evalx = np.array([0,0])
                for i in range(lz):
                    branchz_array = []
                    for r in range(lx):
                        Wz = np.kron(s0,s0)
                        for j in range(lz):
                            UU0z = Uz_array[r][i+j]
                            UU1z = np.conjugate(Uz_array[r][i+j+1]).T
                            Qz = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1z,UU0z)
                            # print(np.linalg.det(Qz)) 
                            uz,sz,vz = np.linalg.svd(Qz) #SVD
                            QQz = np.dot(uz,vz)
                            Wz = np.dot(QQz,Wz)
                        # print(np.linalg.det(Wz)) 
                        evalz, evecz = np.linalg.eig(Wz)
                        evecz = fix(evecz,d)
                        # print(evecz)
                        evalz, evecz = SSort(evalz,evecz)
                        # print(evalz,evecz)
                        if v == 1:
                            branchz = evecz[:,0:2]
                        if v == 2:
                            branchz = evecz[:,2:4]
                        branchz_array.append(branchz)
                    branchz_array.append(branchz_array[0])
                    Wx = s0
                    for e in range(lx):
                        UU0x = Uz_array[e][i]
                        UU1x = np.conjugate(Uz_array[e+1][i]).T
                        qx = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1x,UU0x)
                        Qx = np.dot(np.conjugate(branchz_array[e+1]).T,np.dot(qx,branchz_array[e]))
                        # print(np.dot(np.conjugate(branchz_array[j]).T,branchz_array[j]))
                        ux,sx,vx = np.linalg.svd(Qx) #SVDa
                        QQx = np.dot(ux,vx)
                        # print(np.linalg.det(QQx))
                        Wx = np.dot(QQx,Wx)
                    print("det=",np.round(np.linalg.det(Wx),7),np.round(100*(((q+1+p*ly)/(ly*lt))),2),"%")
                    evalx, evecx = np.linalg.eig(Wx)
                    ea = []
                    for i in range(len(evalx)):
                        ea.append(((1j/(2*np.pi))*llog(evalx[i],0)).real)
                    ea.sort()
                    # print(ea)
                    Evalx = ea+Evalx
                # na1_array[q,p]=(1/(Nz))*Evalx[0]
                # na2_array[q,p]=(1/(Nz))*Evalx[1]
                Pol1.append((1/(Nz))*Evalx[0])
                Pol2.append((1/(Nz))*Evalx[1])
            na1_array.append((1/Ny)*np.sum(Pol1))
            na2_array.append((1/Ny)*np.sum(Pol2))
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
    #     if gap == 0:
    #         ax.scatter(X, Y, na1_array, linewidth=0.1)
    #         ax.scatter(X, Y, na2_array, linewidth=0.1)
    #     if gap == np.pi:
    #         ax.plot_surface(X, Y, na1_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
    #         ax.plot_surface(X, Y, na2_array, rstride=1, cstride=1,linewidth=0.1, cmap='afmhot')
    # ax.set_xlabel('t', fontsize=15, color='black')  
    # ax.set_ylabel('ky', fontsize=15, color='black')  
    # ax.set_zlabel('quasienergy', fontsize=15, color='black')
    # np.savetxt('D:\科研\Topomat\Floquet insulator\\3D-FBOTP\\files\\'+'DP'+str(m)+'-zx-1.txt', np.c_[na1_array],fmt='%.18e',delimiter='\t')
    # np.savetxt('D:\科研\Topomat\Floquet insulator\\3D-FBOTP\\files\\'+'DP'+str(m)+'-zx-2.txt', np.c_[na2_array],fmt='%.18e',delimiter='\t')
    plt.show()

if __name__ == '__main__':
    main()