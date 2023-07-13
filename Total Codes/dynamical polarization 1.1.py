from re import L
import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools
import scipy
from scipy.linalg import expm

J1 = 1
J2 = 2
J3 = 3
dk = 0.001
k_array = list(np.arange(-np.pi,np.pi,dk))
T = (np.pi)/(5)
t_array = np.linspace(0,T,20)
l = len(k_array)
L = len(t_array)

s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

def U(k,t):
    h = (J1+J2*np.cos(k))*s1+J2*np.sin(k)*s2+2*J3*np.sin(k)*s3
    U1 = scipy.linalg.expm(-1j*h*t)
    UF = scipy.linalg.expm(-1j*h*T)
    eigenvalue, eigenvector = np.linalg.eig(UF)
    Eig = np.exp(1j*(-1j*np.log(eigenvalue)-np.pi)*(t/T))
    U2 = np.exp(-1j*np.pi*(t/T))*np.dot(eigenvector,np.dot(np.diag(Eig),np.conjugate(eigenvector).T))
    return np.dot(U1,U2)


def main():
    Q_array = [0 for index in range(l)]
    nu_array = [0 for index in range(L)]
    un_array = [0 for index in range(L)]
    for j in range(L):
        t = t_array[j]
        for i in range(l):
            k = k_array[i]
            UU0 = U(k,t)
            UU1 = np.conjugate(U(k+dk,t)).T
            Q_array[i] = (s0+np.dot(UU1,UU0))/2
        Q_array.reverse()
        W = functools.reduce(lambda x,y:np.dot(x,y),Q_array)
        print(np.linalg.det(W))
        eigenvalue, eigenvector = np.linalg.eig(W)
        nu_array[j] = (1j/(2*np.pi))*np.log(eigenvalue[0])
        un_array[j] = (1j/(2*np.pi))*np.log(eigenvalue[1])
    plt.plot(t_array,nu_array)
    plt.plot(t_array,un_array)
    plt.xlabel("time", fontdict={'size': 16})
    plt.ylabel("polarization", fontdict={'size':16})
    plt.title("dynamical polarization", fontdict={'size': 20})
    plt.show()


if __name__ == '__main__':
    main()