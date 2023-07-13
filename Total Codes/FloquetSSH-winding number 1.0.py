import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools

v=1 #intracell hopping in time period 1
w=1 #intercell hopping in time period 2
aalpha=-0.25 
alpha= math.floor(aalpha) #phase index 1
bbeta=1.25
beta = math.floor(bbeta) #phase index 2
t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
print(t1,t2)
T=t1+t2
Theta = np.pi

dk=0.001 #step
k_array=np.arange(-np.pi,np.pi,dk) #set momentum interval
length = len(k_array) #total steps
print(k_array)

def llog(z, theta):
    argument = np.angle(z) # between -pi and +pi
    modulus = np.abs(z)
    argument = theta - np.mod(theta-argument, 2*np.pi)
    return np.log(modulus)+1j*argument

Phi = [0 for index in range(length)]
for j in range(0,length,1):
    k = k_array[j]
    Phi[j]=(v*t1+w*t2*np.exp(1j*k))

varPhi = [0 for index in range(length)]
for j in range(0,length,1):
    k = k_array[j]
    if t1 >= t2:
        varPhi[j]=(v*(T/2))
    if t1 < t2:
        varPhi[j]=(v*t1+w*(T/2-t1)*np.exp(1j*k)) 

E = [0 for index in range(length)]
for j in range(0,length,1):
    k = k_array[j]
    E[j]=np.abs(Phi[j])/T

varE = [0 for index in range(length)]
for j in range(0,length,1):
    k = k_array[j]
    varE[j]=np.abs(varPhi[j])/T

U = [0 for index in range(length)] 
for j in range(0,length,1):
    k = k_array[j]
    P = np.matrix([[0,np.conjugate(varPhi[j])],[varPhi[j],0]])
    U[j]=scipy.linalg.expm(-1j*P)

trans = [0 for index in range(length)] 
for j in range(0,length,1):
    k = k_array[j]
    trans[j]=(1/(np.sqrt(2)))*np.matrix([[Phi[j]/(E[j]*T),1],[Phi[j]/(E[j]*T),-1]])

transdagger = [0 for index in range(length)] 
for j in range(0,length,1):
    k = k_array[j]
    transdagger[j]=(1/(np.sqrt(2)))*np.matrix([[np.conjugate(Phi[j]/(E[j])*T),np.conjugate(Phi[j]/(E[j])*T)],[1,-1]])

UF = [0 for index in range(length)] 
for j in range(0,length,1):
    k = k_array[j]
    Eigenplus = np.exp(-(1/2)*llog(np.exp(-1j*E[j]*T),Theta))
    Eigenminus = np.exp(-(1/2)*llog(np.exp(1j*E[j]*T),Theta))
    UF[j]=transdagger[j]*np.matrix([[Eigenplus,0],[0,Eigenminus]])*trans[j]

u = [0 for index in range(length)] 
for j in range(0,length,1):
    k = k_array[j]
    UU=U[j]*UF[j]
    eigenvalue, eigenvector = np.linalg.eig(UU)
    u[j]=eigenvalue[0]
    print(eigenvalue)

Tr = [0 for index in range(length)]
for j in range(0,length-1,1):
    Tr[j]=(u[j+1]-u[j])/(u[j])

W=((1j)/(2*np.pi))*functools.reduce(lambda x,y:x+y,Tr)
print(W)