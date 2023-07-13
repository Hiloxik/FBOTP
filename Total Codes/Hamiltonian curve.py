import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

J1 = 1
J2 = 1.1
J = np.linspace(0,20,200)
L =len(J)

figure = plt.figure()
ax = Axes3D(figure)
k = np.linspace(-np.pi,np.pi,500)
for i in range(L):
    J3 = J[i]
    x = (J1+J2*np.cos(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
    y = (J2*np.sin(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
    z = (2*J3*np.sin(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
    ax.plot(x,y,z,label=str(round(J3,1)))


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
my_x_ticks = np.arange(-1, 1, 0.5)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(-1, 1, 0.5)
plt.yticks(my_y_ticks)
ax.set_title('f')
plt.savefig('Hamiltonian curve-'+"J1="+str(J1)+"-"+"J2="+str(J2)+'.jpg', dpi=300)
plt.show()