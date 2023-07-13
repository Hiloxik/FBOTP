import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 

N=27 #number of unit cells
m=13 #length of head or tail

cm = plt.cm.get_cmap('Blues') #get colorbar

#build the OBC Hamiltonian
hamiltonian = np.zeros((2*N,2*N))
for i in range(0,2*m,2):
    hamiltonian[i,i+1] = 1
    hamiltonian[i+1,i] = 1
for i in range(1,2*m,2):
    hamiltonian[i,i+1] = 0
    hamiltonian[i+1,i] = 0
for i in range(2*m,2*(N-m)+1,2):
    if N-2*m-1 == 0:
        hamiltonian[i,i+1] = 1/2
        hamiltonian[i+1,i] = 1/2
    else:
        hamiltonian[i,i+1] = math.cos(((math.pi)/(2*(N-2*m-1)))*(i/2)-(m*math.pi)/(2*(N-2*m-1)))
        hamiltonian[i+1,i] = math.cos(((math.pi)/(2*(N-2*m-1)))*(i/2)-(m*math.pi)/(2*(N-2*m-1)))
for i in range(2*m+1,2*(N-m)+1,2):
    if N-2*m-1 == 0:
        hamiltonian[i,i+1] = 1/2
        hamiltonian[i+1,i] = 1/2
    else:
        hamiltonian[i,i+1] = math.sin(((math.pi)/(2*(N-2*m-1)))*((i-1)/2)-(m*math.pi)/(2*(N-2*m-1)))
        hamiltonian[i+1,i] = math.sin(((math.pi)/(2*(N-2*m-1)))*((i-1)/2)-(m*math.pi)/(2*(N-2*m-1)))
for i in range(2*(N-m),2*N,2):
    hamiltonian[i,i+1] = 0
    hamiltonian[i+1,i] = 0
for i in range(2*(N-m)+1,2*N-1,2):
    hamiltonian[i,i+1] = 1
    hamiltonian[i+1,i] = 1

#solve the eigen-problem
eigenvalue, eigenvector = np.linalg.eig(hamiltonian) 
#divide the eigenvalues from three sectors
eigen1 = eigenvalue[:2*m]
eigen2 = eigenvalue[2*m:2*(N-m)]
eigen3 = eigenvalue[2*(N-m):2*N]


#number the eigenvalues
k1 = np.arange(0,2*m)
k2 = np.arange(2*m,2*(N-m))
k3 = np.arange(2*(N-m),2*N)
k = np.arange(0,2*N)
z = k2

#plot the spectrum
plt.scatter(k1,eigen1,c=seaborn.xkcd_rgb['dark indigo'],marker = "o",s=30,alpha=1,norm=1)
plt.scatter(k2,eigen2,c=z, marker = "o",s=30,alpha=1)
plt.scatter(k3,eigen3,c=seaborn.xkcd_rgb['bright yellow'],marker = "o",s=30,alpha=1,norm=1)
plt.colorbar()
plt.xlabel("eigenstate", fontdict={'size': 16})
plt.ylabel("energy", fontdict={'size':16})
plt.title("OBC-"+str(N)+"cells-"+str(m)+"flat", fontdict={'size': 20})
plt.savefig('OBC-'+str(N)+"-"+str(m)+'.jpg', dpi=300)
plt.show()