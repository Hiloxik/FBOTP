import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm

#parameters
T = 2
J1 = (np.pi/2)/(np.sqrt(2))
J2 = (3*np.pi/4)/(np.sqrt(2))
d = 4
nx = 1
ny = 1
N = 30
kx_array = np.linspace(-np.pi,np.pi,N+1)
ky_array = np.linspace(-np.pi,np.pi,N+1)
gap = 0
lx = len(kx_array)-1
ly = len(ky_array)-1

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

#anomalous periodic operator
def U(kx,ky,t):
    #Hamiltonians
    H1 = (2*J1/T)*(np.kron(s1,s0)-np.kron(s2,s2))+(0*2/T)*(np.sin(ky)*np.kron(s3,s1)+np.cos(ky)*np.kron(s3,s2)-0.4*np.cos(ky)*np.kron(s0,s0))+(0.0001)*(np.kron(s3,s0)+0.5*np.kron(s3,s3))
    H2 = (2*J2/T)*(np.cos(nx*kx)*np.kron(s1,s0)-np.sin(nx*kx)*np.kron(s2,s3)-np.cos(ny*ky)*np.kron(s2,s2)-np.sin(ny*ky)*np.kron(s2,s1))
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/4)),np.dot(expm(-1j*H2*(T/2)),expm(-1j*H1*(T/4))))
    #sort by quasienergies
    eval,evec = np.linalg.eig(UF)
    eval,evec = ssort(eval,evec)
    #fix phase
    evec = fix(evec,d)

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

#polarization calculation
def main():
    nu1_array = [0 for index in range(lx+1)]
    nu2_array = [0 for index in range(lx+1)]
    nu3_array = [0 for index in range(lx+1)]
    nu4_array = [0 for index in range(lx+1)]
    t = T/2
    for i in range(lx):
        kx = kx_array[i]
        W = np.kron(s0,s0)
        ky = -np.pi
        for j in range(ly):
            dk = j*((2*np.pi)/(N))
            UU0 = U(kx,ky+dk,t)
            UU1 = np.conjugate(U(kx,ky+dk+(2*np.pi)/(N),t)).T
            Q = (1-1/(2*(1)))*np.kron(s0,s0)+(1/(2*(1)))*np.dot(UU1,UU0)
            u,s,v = np.linalg.svd(Q) #SVD
            QQ = np.dot(u,v)
            W = np.dot(QQ,W) 
        print(np.linalg.det(W),i)
        eigenvalue, eigenvector = np.linalg.eig(W)
        ee = [0 for index in range(len(eigenvalue))]
        for i in range(len(eigenvalue)):
            ee[i] = (1j/(2*np.pi))*llog(eigenvalue[i],0)
        ee.sort()
        nu1_array[j] = ee[0]
        nu2_array[j] = ee[1]
        nu3_array[j] = ee[2]
        nu4_array[j] = ee[3]
    # tt = np.linspace(0,T,100)
    kk = np.linspace(-np.pi,np.pi,100)
    z1 = 0*kk+0.5
    z2 = 0*kk+1
    z3 = 0*kk+0
    plt.scatter(kx_array,nu1_array)
    plt.scatter(kx_array,nu2_array)
    plt.scatter(kx_array,nu3_array)
    plt.scatter(kx_array,nu4_array)
    plt.plot(kk,z1,c='black')
    plt.plot(kk,z2,c='black')
    plt.plot(kk,z3,c='black')
    plt.xlabel("kx", fontdict={'size': 16})
    plt.ylabel("polarization", fontdict={'size':16})
    plt.title("dynamical polarization", fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()