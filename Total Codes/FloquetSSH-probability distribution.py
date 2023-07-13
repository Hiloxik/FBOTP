import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools

N=30 #number of unit cells
v=1 #intracell hopping in time period 1
w=1 #intercell hopping in time period 2
aalpha=0.25 
alpha= math.floor(aalpha) #phase index 1
bbeta=1.25
beta = math.floor(bbeta) #phase index 2
t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
print(t1,t2)

dt=0.0001 #step
t_array=np.arange(0,t1+t2,dt) #set time interval
length = len(t_array) #total steps

cm = plt.cm.get_cmap('plasma') #get colorbar

#get interaction parameters during one period
V = [0 for index in range(length)] #intercell hopping array
for i in range(0,length,1):
    t = t_array[i]
    if 0 <= t <= t1:
        V[i]=v
    if t1 < t <= t1+t2:
        V[i]=0 
W = [0 for index in range(length)] #intracell hopping array
for i in range(0,length,1):
    t = t_array[i]
    if 0 <= t <= t1:
        W[i]=0
    if t1 < t <= t1+t2:
        W[i]=w

#calculate the evolution operator
evol = [0 for index in range(length)] 
for j in range(0,length,1):
    h = np.zeros((2*N,2*N)) #hamiltonian in one step
    for i in range(0,2*N,2):
        h[i,i+1] = V[j]*dt
        h[i+1,i] = V[j]*dt
    for i in range(1,2*N-1,2):
        h[i,i+1] = W[j]*dt
        h[i+1,i] = W[j]*dt
    evol[j] = h
evol.reverse()

#total Floquet operator
Floquetian = scipy.linalg.expm(-1*1j*functools.reduce(lambda x,y:x+y,evol))
# print(np.round(Floquetian,2))

#solve the eigen-problem
eigenvalue, eigenvector = np.linalg.eig(Floquetian)

prob = [0 for index in range(2*N)] 
for i in range(2*N):
    brai=np.zeros((1,2*N))
    brai[0,i]=1
    prob[i]=math.pow(np.abs(np.dot(brai,eigenvector[:,i])),2)

k = np.arange(0,2*N)

plt.plot(k, prob, c="red")
plt.xlabel("eigenstate", fontdict={'size': 16})
plt.ylabel("probability", fontdict={'size':16})
plt.title("prob-"+str(N)+"cells-"+"a="+str(round(aalpha,2))+"-b="+str(round(bbeta,2)), fontdict={'size': 20})
# plt.savefig('prob-'+str(N)+"-"+str(round(aalpha,2))+"-"+str(round(bbeta,2))+'.jpg', dpi=300)
plt.show()