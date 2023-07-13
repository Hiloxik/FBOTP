import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools
from scipy.linalg import expm

cm = plt.cm.get_cmap('plasma')

#parameters
T = 20
J1 = (3*np.pi/4)/(np.sqrt(2))
J = np.linspace(0,(2*np.pi)/(np.sqrt(2)),50)
d = 4
n = 4
Nx = 10
Ny = 10
v = 1
kx_array = np.linspace(-np.pi,np.pi,Nx+1)
ky_array = np.linspace(-np.pi,np.pi,Nx+1)
t_array = np.linspace(0.00001*T,0.99999*T,21)
gap = 0
lx = len(kx_array)
ly = len(ky_array)
l = len(J)

#pauli matrix
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

#Hamiltonian
def H(kx,ky,J2):
    H1 = (2*J1/T)*(np.kron(s1,s0)-np.kron(s2,s2))+(0*2*J1/T)*np.kron(s0,s3)+(0*2*J1/T)*np.cos(n*ky)*np.kron(s1,s1)+(0*2/T)*np.kron(s3,s0)+(0*2/T)*(np.sin(n*ky)*np.kron(s3,s1)+np.cos(n*ky)*np.kron(s3,s2)-0.4*np.cos(n*ky)*np.kron(s0,s0))
    H2 = (2*J2/T)*(np.cos(kx)*np.kron(s1,s0)-np.sin(kx)*np.kron(s2,s3)-np.cos(n*ky)*np.kron(s2,s2)-np.sin(n*ky)*np.kron(s2,s1))+(0*J1/T)*np.kron(s0,s3)+(0*2/T)*np.kron(s3,s0)+(0*2*J1/T)*np.cos(n*ky)*np.kron(s1,s1)
    #Floquet operator
    UF = np.dot(expm(-1j*H1*(T/4)),np.dot(expm(-1j*H2*(T/2)),expm(-1j*H1*(T/4))))
    return UF

#draw quasienergy spectrum
def main():
    for i in range(ly):
        ky = ky_array[i]
        for j in range(lx):
            kx = kx_array[j]
            zero = [0 for index in range(l)]
            zerominus = [0 for index in range(l)]
            for i in range(l):
                J2 = J[i]
                eigenvalue, eigenvector = np.linalg.eig(H(kx,ky,J2))
                eigenvalue.sort()
                zero[i] = (-1j*np.log(eigenvalue[1])).real
                zerominus[i] = -zero[i]
            plt.scatter(J, zerominus, c=zerominus, cmap=cm)
            plt.scatter(J, zero, c=zero, cmap=cm)
        y = 0*J
        y1 = 0*J+np.pi
        y2 = 0*J-np.pi
        plt.plot(J,y,c='black')
        plt.plot(J,y1,c='black')
        plt.plot(J,y2,c='black')
        plt.xlabel("J2", fontdict={'size': 16})
        plt.ylabel("quasienergy", fontdict={'size':16})
        plt.title("PBC", fontdict={'size': 20})
        plt.show()

if __name__ == '__main__':
    main()