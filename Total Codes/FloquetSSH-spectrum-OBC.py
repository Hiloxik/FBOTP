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
aalpha=-0.5 
alpha= math.floor(aalpha) #phase index 1
bbeta=1.5
beta = math.floor(bbeta) #phase index 2
t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
print(t1,t2)

dt=0.001 #step
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
    for i in range(0,2*N-1,2):
        h[i,i+1] = V[j]*dt
        h[i+1,i] = V[j]*dt
    for i in range(1,2*N-1,2):
        h[i,i+1] = W[j]*dt
        h[i+1,i] = W[j]*dt
    evol[j] = scipy.linalg.expm(-1*1j*h)
evol.reverse()

#total Floquet operator
Floquetian = functools.reduce(lambda x,y:np.dot(x,y),evol)
# print(np.round(Floquetian,2))

#solve the eigen-problem
eigenvalue, eigenvector = np.linalg.eig(Floquetian)
quasienergy = [0 for index in range(2*N)]
for i in range(0,2*N,1):
    quasienergy[i] = (cmath.phase(eigenvalue[i])/(t1+t2))

quasienergy.sort()

#number the eigenvalues
k = np.arange(0,2*N)
z = quasienergy
x=np.arange(0,2*N+0.1,0.1)
y0=0*x
y1=0*x+(np.pi)/(t1+t2)
y2=0*x-(np.pi)/(t1+t2)

#draw the spectrum
plt.scatter(k, quasienergy, c=z, s=30, cmap=cm)
plt.plot(x, y0, color="steelblue", alpha=0.6)
plt.plot(x, y1, color="steelblue", alpha=0.6)
plt.plot(x, y2, color="steelblue", alpha=0.6)
plt.colorbar()
plt.xlabel("eigenstate", fontdict={'size': 16})
plt.ylabel("quasienergy", fontdict={'size':16})
my_y_ticks = np.arange(-(np.pi)/(t1+t2), (np.pi)/(t1+t2)+(np.pi)/(4*(t1+t2)), (np.pi)/(4*(t1+t2)))
plt.yticks(my_y_ticks)
plt.title("OBC-"+str(N)+"cells-"+"a="+str(round(aalpha,2))+"-b="+str(round(bbeta,2)), fontdict={'size': 20})
# plt.savefig('OBC-'+str(N)+"-"+str(round(aalpha,2))+"-"+str(round(bbeta,2))+'.jpg', dpi=300)
plt.show()