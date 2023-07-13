import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools

cm = plt.cm.get_cmap('plasma')

J1=1 #intercell hopping
J2=2 #betweencell hopping
J3 = np.linspace(0,5,200)
l = len(J3)
N=30 #number of unit cells

def H(J):
    hamiltonian = np.zeros((2*N,2*N),dtype=complex)
    for i in range(0,2*N,2):
        hamiltonian[i,i+1] = J1
        hamiltonian[i+1,i] = J1
    for i in range(0,2*N-2,2):
        hamiltonian[i,i+2] = -1j*J
        hamiltonian[i+2,i] = 1j*J
    for i in range(1,2*N-1,2):
        hamiltonian[i,i+1] = J2
        hamiltonian[i+1,i] = J2
    for i in range(1,2*N-2,2):
        hamiltonian[i,i+2] = 1j*J
        hamiltonian[i+2,i] = -1j*J
    return hamiltonian
# print(hamiltonian)

def main():
    for j in range(2*N):
        zero = [0 for index in range(l)]
        for i in range(l):
            J = J3[i]
            eigenvalue, eigenvector = np.linalg.eig(H(J))
            eigenvalue.sort()
            zero[i] = eigenvalue[j].real
            print(zero[i])
        plt.plot(J3, zero)
    y = 0*J3
    plt.plot(J3,y,c='black')
    plt.xlabel("J3", fontdict={'size': 16})
    plt.ylabel("energy", fontdict={'size':16})
    plt.title('OBC-'+str(N)+"cells-"+"J1="+str(J1)+"-"+"J2="+str(J2), fontdict={'size': 20})
    plt.savefig('OBC-'+str(N)+"cells-"+"J1="+str(J1)+"-"+"J2="+str(J2)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()