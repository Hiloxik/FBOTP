import numpy as np
import matplotlib.pyplot as plt 
import cmath 

cm = plt.cm.get_cmap('plasma')

J1=0.2 #intercell hopping
J2=1 #betweencell hopping
J3 = 1
N=30 #number of unit cells

#build the OBC Hamiltonian
hamiltonian = np.zeros((2*N,2*N),dtype=complex)
for i in range(0,2*N,2):
    hamiltonian[i,i+1] = J1
    hamiltonian[i+1,i] = J1
for i in range(0,2*N-2,2):
    hamiltonian[i,i+2] = -1j*J3
    hamiltonian[i+2,i] = 1j*J3
for i in range(1,2*N-1,2):
    hamiltonian[i,i+1] = J2
    hamiltonian[i+1,i] = J2
for i in range(1,2*N-2,2):
    hamiltonian[i,i+2] = 1j*J3
    hamiltonian[i+2,i] = -1j*J3
# print(hamiltonian)

    
#solve the eigen-problem
eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
# eigenvalue.sort()

k = np.arange(0,2*N) 
z = eigenvalue.real
F = 0*k

#draw the spectrum
plt.scatter(k, z,c=z,cmap=cm)
plt.plot(k,F)
plt.colorbar()
plt.xlabel("eigenstate", fontdict={'size': 16})
plt.ylabel("energy", fontdict={'size':16})
plt.title("OBC", fontdict={'size': 20})
plt.show()