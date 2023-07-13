import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools

t1=1
t2=2
k=3
A=np.mat([[0,t1],[t1,0]])
B=np.mat([[0,t2*np.exp(-1j*k)],[t2*np.exp(1j*k),0]])
C=np.dot(scipy.linalg.expm(-1j*B),scipy.linalg.expm(-1j*A))
D=np.mat([[np.cos(t1)*np.cos(t2)-np.sin(t1)*np.sin(t2)*np.exp(-1j*k),-1j*np.sin(t1)*np.cos(t2)-1j*np.cos(t1)*np.sin(t2)*np.exp(-1j*k)],[-1j*np.sin(t1)*np.cos(t2)-1j*np.cos(t1)*np.sin(t2)*np.exp(1j*k),np.cos(t1)*np.cos(t2)-np.sin(t1)*np.sin(t2)*np.exp(1j*k)]])

C1=scipy.linalg.expm(-1j*A)
D1=np.mat([[np.cos(t1),-1j*np.sin(t1)],[-1j*np.sin(t1),np.cos(t1)]])

C2=scipy.linalg.expm(-1j*B)
D2=np.mat([[np.cos(t2),-1j*np.exp(-1j*k)*np.sin(t2)],[-1j*np.exp(1j*k)*np.sin(t2),np.cos(t2)]])
print(C-D)
