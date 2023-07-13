import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools

cm = plt.cm.get_cmap('plasma')

#parameters
t1= 2*np.pi
t = np.linspace(0,2*np.pi,200)
n = 1
l = len(t)
K = np.arange(-np.pi,np.pi,0.05*np.pi)
L = len(K)

#pauli matrix
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

#Hamiltonian
def H(k,t2):
    a0 = np.cos(t1)*np.cos(t2)-np.sin(t1)*np.sin(t2)*np.cos(n*k)
    a1 = np.cos(t1)*np.sin(t2)*np.cos(n*k)+np.sin(t1)*np.cos(t2)
    a2 = np.cos(t1)*np.sin(t2)*np.sin(n*k)
    a3 = -np.sin(t1)*np.sin(t2)*np.sin(n*k)
    hamiltonian = (a0*s0+1j*(a1*s1+a2*s2+a3*s3))
    return hamiltonian

#draw quasienergy spectrum
def main():
    for j in range(L):
        k = K[j]
        zero = [0 for index in range(l)]
        zerominus = [0 for index in range(l)]
        for i in range(l):
            t2 = t[i]
            eigenvalue, eigenvector = np.linalg.eig(H(k,t2))
            eigenvalue.sort()
            zero[i] = (-1j*np.log(eigenvalue[1])).real
            zerominus[i] = -zero[i]
        plt.scatter(t/np.pi, zerominus, c=zerominus, cmap=cm)
        plt.scatter(t/np.pi, zero, c=zero, cmap=cm)
    y = 0*t
    plt.plot(t/np.pi,y,c='black')
    plt.xlabel("t2", fontdict={'size': 16})
    plt.ylabel("energy", fontdict={'size':16})
    plt.title("PBC", fontdict={'size': 20})
    plt.show()

if __name__ == '__main__':
    main()