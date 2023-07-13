import numpy as np
import matplotlib.pyplot as plt 
import cmath 

v=1 #intercell hopping
w=0.2 #betweencell hopping
N=8 #number of unit cells

#build the OBC Hamiltonian
hamiltonian = np.zeros((2*N,2*N))
for i in range(0,2*N,2):
    hamiltonian[i,i+1] = v
    hamiltonian[i+1,i] = v
for i in range(1,2*N-1,2):
    hamiltonian[i,i+1] = w
    hamiltonian[i+1,i] = w
hamiltonian[0,2*N-1] = w
hamiltonian[2*N-1,0] = w
    
#solve the eigen-problem
eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
eigenvalue.sort()

k = np.arange(0,2*N) 

#draw the spectrum
plt.scatter(k, eigenvalue)
plt.xlabel("eigenstate", fontdict={'size': 16})
plt.ylabel("energy", fontdict={'size':16})
plt.title("PBC", fontdict={'size': 20})
plt.show()