import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

w=np.linspace(0,10,100)
length=len(w)

# k_array=np.linspace(-np.pi,np.pi,2)
# length=len(W)
Q = [0 for index in range(length)]
for i in range(0,length,1):
    W=w[i]
    E = cmath.phase(np.cos(W+2)*np.cos(W)-np.sin(W+2)*np.sin(W)*np.cos(0)+1j*np.sqrt(1-math.pow((np.cos(W+2)*np.cos(W)-np.sin(W+2)*np.sin(W)*np.cos(0)),2)))
    Q[i] = E
P = [0 for index in range(length)]
for i in range(0,length,1):
    W=w[i]
    F = cmath.phase(np.cos(W+2)*np.cos(W)-np.sin(W+2)*np.sin(W)*np.cos(np.pi)+1j*np.sqrt(1-math.pow((np.cos(W+2)*np.cos(W)-np.sin(W+2)*np.sin(W)*np.cos(np.pi)),2)))
    P[i] = F
plt.plot(w, P, color="red", alpha=0.6)
plt.plot(w, Q, color="steelblue", alpha=0.6)
plt.show()
