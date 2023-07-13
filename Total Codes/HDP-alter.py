import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools
from scipy.linalg import expm

v = 1.6*np.pi
w = -0.5*np.pi
t1 = (v+w)/2
t2 = (v-w)/2
print(t1,t2)
T = t1+t2
n = 4
N = 10000
k_array = np.linspace(-np.pi,np.pi,N)
t_array = np.linspace(0,T,100)
gap = np.pi
l = len(k_array)
L = len(t_array)

s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j*argument

def U(k,t):
    H1 = s1
    H2 = np.cos(n*k)*s1+np.sin(n*k)*s2
    UF = np.dot(expm(-1j*t2*H2),expm(-1j*t1*H1))
    Eigenvalue, Eigenvector = np.linalg.eig(UF)
    Eigval = np.abs(Eigenvalue[1])
    sorted_indices = np.argsort(Eigenvalue)
    Eigvec0 = Eigenvector[:,sorted_indices[:-1-1:-1]]
    Eigvec1 = Eigenvector[:,sorted_indices[:-2-1:-1]]
    
    if t <= t1:
        U1 = np.cos(t)*s0-1j*np.sin(t)*s1
    if t1 < t <= T:
        U1 = np.dot(np.cos(t-t1)*s0-1j*np.sin(t-t1)*np.cos(n*k)*s1-1j*np.sin(t-t1)*np.sin(n*k)*s2,np.cos(t1)*s0-1j*np.sin(t1)*s1)
    
    if gap == 0:
        U2 = np.exp(-1j*Eigval*(t/T))*(np.dot(Eigvec0,Eigvec0.T)+np.dot(Eigvec1,Eigvec1.T))+np.exp(-1j*(2*np.pi-Eigval)*(t/T))*(np.dot(Eigvec0,Eigvec0.T)+np.dot(Eigvec1,Eigvec1.T))
    if gap == np.pi:
        U2 = np.exp(-1j*Eigval*(t/T))*(np.dot(Eigvec0,Eigvec0.T)+np.dot(Eigvec1,Eigvec1.T))+np.exp(-1j*(-Eigval)*(t/T))*(np.dot(Eigvec0,Eigvec0.T)+np.dot(Eigvec1,Eigvec1.T))
    return np.dot(U1,U2)

def main():
    nu_array = [0 for index in range(L)]
    un_array = [0 for index in range(L)]
    for j in range(L):
        t = t_array[j]
        W = s0
        for k in k_array:
            UU0 = U(k,t)
            UU1 = np.conjugate(U(k+(2*np.pi/N),t)).T
            Q = (1-1/(2*(n)))*s0+(1/(2*(n)))*np.dot(UU1,UU0)
            W = np.dot(Q,W) 
        print(np.linalg.det(W),j)
        eigenvalue, eigenvector = np.linalg.eig(W)
        nu_array[j] = (1j/(2*np.pi))*llog(eigenvalue[0],0)
        un_array[j] = (1j/(2*np.pi))*llog(eigenvalue[1],0)
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