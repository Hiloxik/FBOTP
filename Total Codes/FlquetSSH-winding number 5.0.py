import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

v=1 #intracell hopping in time period 1
w=1 #intercell hopping in time period 2
aalpha=0.25 #phase index 1
bbeta=1.5 #phase index 2
t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
print(t1,t2)
T=t1+t2 #total time period

V=v*t1
W=w*t1

# V=np.pi-0.2
# W=np.linspace(0,np.pi,50)
# length=len(W)

sig0 = np.matrix([[1,0],[0,1]])
sig1 = np.matrix([[0,1],[1,0]])
sig2 = np.matrix([[0,-1j],[1j,0]])
sig3 = np.matrix([[1,0],[0,-1]])

def nU(k):
    nU0 = np.cos(W/2)*np.cos(V/2)-np.sin(W/2)*np.sin(V/2)*np.cos(k)
    nU1 = -1j*(np.cos(W/2)*np.sin(V/2)+np.sin(W/2)*np.cos(V/2)*np.cos(k))
    nU2 = -1j*(np.sin(W/2)*np.cos(V/2)*np.sin(k))
    nU3 = 1j*(np.sin(W/2)*np.sin(V/2)*np.sin(k))
    return np.dot(nU0,sig0)+np.dot(nU1,sig1)+np.dot(nU2,sig2)+np.dot(nU3,sig3)

# def aU(k):
#     t0 = np.cos(V)*np.cos(W)-np.sin(V)*np.sin(W)*np.cos(k)
#     t1 = -np.sin(V)*np.cos(W)-np.cos(V)*np.sin(W)*np.cos(k)
#     t2 = -np.sin(W)*np.sin(k)
#     aU0 = np.sqrt((1+t0)/2)
#     aU1 = -1j*np.sqrt(1/(2*(1+t0)))*t1
#     aU2 = -1j*np.sqrt(1/(2*(1+t0)))*t2
#     return np.dot(aU0,sig0)+np.dot(aU1,sig1)+np.dot(aU2,sig2)

def aU(k):
    t0 = np.cos(V)*np.cos(W)-np.sin(V)*np.sin(W)*np.cos(k)
    t1 = -np.sin(V)*np.cos(W)-np.cos(V)*np.sin(W)*np.cos(k)
    t2 = -np.sin(W)*np.sin(k)
    aU0 = -1j*np.sqrt((1-t0)/2)
    aU1 = np.sqrt(1/(2*(1-t0)))*t1
    aU2 = np.sqrt(1/(2*(1+t0)))*t2
    return np.dot(aU0,sig0)+np.dot(aU1,sig1)+np.dot(aU2,sig2)

# def U(k):
#     t0 = np.cos(V)*np.cos(W)-np.sin(V)*np.sin(W)*np.cos(k)
#     t1 = -np.sin(V)*np.cos(W)-np.cos(V)*np.sin(W)*np.cos(k)
#     t2 = -np.sin(W)*np.sin(k)
#     aU0 = np.sqrt((1+t0)/2)
#     aU1 = -1j*np.sqrt(1/(2*(1+t0)))*t1
#     aU2 = -1j*np.sqrt(1/(2*(1+t0)))*t2
#     U0 = np.cos((V+W)/2)*aU0-np.cos(k)*np.sin((V+W)/2)*aU1-np.sin(k)*np.sin((V+W)/2)*aU2
#     U1 = np.cos((V+W)/2)*aU1+np.sin((V+W)/2)*np.cos(k)*aU0
#     U2 = np.cos((V+W)/2)*aU2+np.sin((V+W)/2)*np.sin(k)*aU0
#     U3 = np.cos(k)*np.sin((V+W)/2)*aU2-np.sin(k)*np.sin((V+W)/2)*aU1
#     return U0*sig0-1j*U1*sig1-1j*U2*sig2-1j*U3*sig3

#chiral matrix
def S(k):
    diag,P=np.linalg.eig(np.dot(nU(k),aU(k))) #eigen-problem of anamolous time operator at half period
    Trans0=np.matrix([[1,0],[0,1]])
    Trans1=np.matrix([[1,0],[0,-1]])
    Trans2=(1/np.sqrt(2))*np.matrix([[1,1],[1,-1]])
    S=np.dot(P,np.dot(Trans1,np.dot(Trans2,scipy.linalg.inv(P))))
    return np.round((1/2)*(Trans0-S),2)

def main():
    delta_1 = 1e-9 #differentiation step
    delta_2 = 1e-3 #integration step
    # X = [0 for index in range(length)]
    # for i in range(0,length,1):
    #     w=W[i]
    C=0
    for k in np.arange(-np.pi, np.pi, delta_2):
        Ph0=np.dot(nU(k),aU(k))
        Ph1=np.dot(nU(k+delta_1),aU(k+delta_1))
        C=C+np.trace(np.dot(S(k),np.dot(scipy.linalg.inv(Ph0),((Ph1-Ph0)/delta_1))))*delta_2
    # X[i]=C/(2*np.pi)
    # plt.scatter(W,X,color="steelblue", alpha=0.6)
    # plt.xlabel("W", fontdict={'size': 16})
    # plt.ylabel("winding number", fontdict={'size':16})
    # # my_y_ticks = np.arange(-0.1, 1.1, 0.1)
    # # plt.yticks(my_y_ticks)
    # plt.show()
    print('Winding number = ', 1j*C/(2*np.pi))

if __name__ == '__main__':
    main()