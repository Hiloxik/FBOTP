import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

#parameter settings
# v=1 #intracell hopping in time period 1
# w=1 #intercell hopping in time period 2
# aalpha=0.25 #phase index 1
# bbeta=4.75 #phase index 2
# t1=(np.pi*(aalpha+bbeta))/(2*v) #time period 1
# t2=(np.pi*(bbeta-aalpha))/(2*w) #time period 2 
# print(t1,t2)
# T=t1+t2 #total time period

# def f(k):
#     f = np.cos(V)*np.cos(W)-np.sin(V)*np.sin(W)*np.cos(k)
#     return f

# def gx(k):
#     gx = -np.sin(V)*np.cos(W)-np.cos(V)*np.sin(W)*np.cos(k)
#     return gx

# def gy(k):
#     gy = -np.cos(V)*np.sin(W)*np.sin(k)
#     return gy

# def gz(k):
#     gz = np.sin(V)*np.sin(W)*np.sin(k)
#     return gz

# def phase(k):
#     phase = np.log((gx(k)+1j*gy(k))/np.sqrt(math.pow(gx(k),2)+math.pow(gy(k),2)))
#     return phase

W=np.linspace(0,2*np.pi,200)
length=len(W)

def F(W,k):
    F = 1+(np.sin(k))/np.sqrt(math.pow(1/(np.tan(W+1.1*(np.pi))),2)+math.pow(1/(np.tan(W)),2)+math.pow(np.sin(k),2)+2*(1/(np.tan(W+1.1*(np.pi))))*(1/(np.tan(W)))*np.sin(k))
    return F

def G(W,k):
    G = (np.tan(W+1.1*(np.pi))*(1/(np.tan(W)))*np.cos(k)+1)/(math.pow(np.tan(W+1.1*(np.pi))*(1/(np.tan(W))),2)+2*np.tan(W+1.1*(np.pi))*(1/(np.tan(W)))*np.cos(k)+1)
    return G

def main():
    delta_1 = 1e-9 #differentiation step
    delta_2 = 1e-3 #integration step
    X = [0 for index in range(length)]
    for i in range(0,length,1):
        w=W[i]
        C=0
        for k in np.arange(-np.pi, np.pi, delta_2):
            Ph0=F(w,k)*G(w,k)
            C=C+Ph0*delta_2
        X[i]=C/(2*np.pi)
    plt.plot(W,X,color="steelblue", alpha=0.6)
    plt.xlabel("W", fontdict={'size': 16})
    plt.ylabel("winding number", fontdict={'size':16})
    my_y_ticks = np.arange(-0.1, 1.1, 0.1)
    plt.yticks(my_y_ticks)
    plt.show()
    # print('Winding number = ', C/(2*np.pi))


if __name__ == '__main__':
    main()