import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools

cm = plt.cm.get_cmap('plasma')

J1=2 #intercell hopping
J2=1 #betweencell hopping
J3 = np.linspace(-20,20,200)
l = len(J3)
K = np.arange(0,2*np.pi,0.05*np.pi)
L = len(K)

def H(k,J):
    hamiltonian = np.zeros((2,2),dtype=complex)
    hamiltonian[0,0] = 2*J*np.sin(k)
    hamiltonian[0,1] = J1+J2*np.exp(-1j*k)
    hamiltonian[1,0] = J1+J2*np.exp(1j*k)
    hamiltonian[1,1] = -2*J*np.sin(k)
    return hamiltonian
# print(hamiltonian)

# def median(data):
#     data.sort()
#     half = len(data)//2
#     return (data[half]+data[~half])/2

def main():
    for j in range(L):
        k = K[j]
        zero = [0 for index in range(l)]
        zerominus = [0 for index in range(l)]
        for i in range(l):
            J = J3[i]
            eigenvalue, eigenvector = np.linalg.eig(H(k,J))
            eigenvalue.sort()
            zero[i] = np.abs(eigenvalue[1].real)
            zerominus[i] = -zero[i]
        plt.plot(J3,zerominus)
        plt.plot(J3, zero)
    # y = 0*J3
    # plt.plot(J3,y,c='black')
    plt.xlabel("J3", fontdict={'size': 16})
    plt.ylabel("energy", fontdict={'size':16})
    plt.title("PBC", fontdict={'size': 20})
    plt.show()

if __name__ == '__main__':
    main()