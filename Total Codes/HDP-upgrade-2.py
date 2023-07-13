import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools
from scipy.linalg import expm

#parameters
t1 = 0.5*np.pi
t2 = np.pi
print(t1,t2)
T = t1+t2
n = 2
N = 200
k_array = np.linspace(-np.pi,np.pi,N+1)
t_array = np.linspace(0,T,30)
gap = 0
l = len(k_array)
L = len(t_array)

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

#anomalous periodic operator
def U(k,t):
    #Hamiltonians
    H1 = s1
    H2 = np.cos(n*k)*s1+np.sin(n*k)*s2
    #Floquet operator
    UF = np.dot(expm(-1j*H2*t2),expm(-1j*H1*t1))
    #sort by quasienergies
    eval,Evec = np.linalg.eig(UF)
    e = -1j*np.log(eval)
    eval_sort_idx = np.argsort(e)[::-1]
    e = np.sort(e)[::-1]
    Eval = [0 for index in range(len(eval))]
    for i in range(len(eval)):
        Eval[i] = np.exp(1j*e[i])
    Evec = Evec[:,eval_sort_idx]
    #fix phase
    phase1 = cmath.phase(Evec[0,0])
    phase2 = cmath.phase(Evec[0,1])
    Phase = np.matrix([[np.exp(-1j*phase1),0],[0,np.exp(-1j*phase2)]])
    Evec = np.dot(Evec,Phase)

    #evolution operator
    r = int(t//T)

    if 0 <= t%T < t1:
        U1 = np.dot(expm(-1j*H1*(t%T)),np.linalg.matrix_power(UF,r))
    if t1 <= t%T < T:
        U1 = np.dot(expm(-1j*H2*(t%T-t1)),np.dot(np.cos(t1)*s0-1j*np.sin(t1)*s1,np.linalg.matrix_power(UF,r)))
    
    #anomalous periodic operator for two gaps
    if gap == 0:
        Eaval = [0 for index in range(len(Eval))]
        for i in range(len(Eval)):
            Eaval[i] = np.exp(-llog(Eval[i],0)*(t/T))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(Evec, np.dot(Smatpi, np.linalg.inv(Evec)))
    if gap == np.pi:
        Eaval = [0 for index in range(len(Eval))]
        for i in range(len(Eval)):
            Eaval[i] = np.exp(-llog(Eval[i],np.pi)*(t/T))
        Smatpi = np.diag(Eaval)
        U2 = np.dot(Evec, np.dot(Smatpi, np.linalg.inv(Evec)))
    return np.dot(U1,U2)

#polarization calculation
def main():
    nu_array = [0 for index in range(L)]
    un_array = [0 for index in range(L)]
    for j in range(L):
        t = t_array[j]
        W = s0
        for i in range(l-1):
            k = k_array[i]
            UU0 = U(k,t)
            # print(U(k,t)-U(k,t+2*T))
            UU1 = np.conjugate(U(k+(2*np.pi/N),t)).T
            Q = (1-1/(2*(n)))*s0+(1/(2*(n)))*np.dot(UU1,UU0)
            u,s,v = np.linalg.svd(Q) #SVD
            QQ = np.dot(u,v)
            W = np.dot(QQ,W) 
        print(np.linalg.det(W),j)
        eigenvalue, eigenvector = np.linalg.eig(W)
        ee = [0 for index in range(len(eigenvalue))]
        for i in range(len(eigenvalue)):
            ee[i] = (1j/(2*np.pi))*llog(eigenvalue[i],0)
        ee.sort()
        nu_array[j] = ee[0]
        un_array[j] = ee[1]
    tt = np.linspace(0,T,100)
    z1 = 0*tt+0.5
    plt.scatter(t_array,nu_array)
    plt.scatter(t_array,un_array)
    plt.plot(tt,z1,c='black')
    plt.xlabel("time", fontdict={'size': 16})
    plt.ylabel("polarization", fontdict={'size':16})
    plt.title("dynamical polarization", fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()