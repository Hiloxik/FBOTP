import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools

cm = plt.cm.get_cmap('plasma') #get colorbar

v=1 #intracell hopping in time period 1
w=1 #intercell hopping in time period 2
alpha=1.5 #phase index 1 = vt1/wt2
beta=5 #phase index 2 = (v*t1+w*t2)/(pi)
t1=(alpha*beta*np.pi)/(v*(alpha+1)) #time period 1
t2=(beta*np.pi)/(w*(alpha+1)) #time period 2 

k = np.arange(-np.pi,np.pi+0.11,0.1)
E = np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)

interval0 = [1 if (i<=np.pi) else 0 for i in E]
interval1 = [1 if (i>=np.pi) else 0 for i in E]

if 0<= beta <= 1:
    E1 = np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)
if 1 < beta <= 3:
    E1 = np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)*interval0+(2*np.pi-np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval1
if 3 < beta <= 5:
    E1 = np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)*interval0+(2*np.pi-np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval0+(4*np.pi-np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval1

if 0<= beta <= 1:
    E2 = -np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)
if 1 < beta <=3:
    E2 = -np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)*interval0+(-2*np.pi+np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval1
if 3 < beta <=5:
    E2 = -np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2)*interval0+(-2*np.pi+np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval0+(-4*np.pi+np.sqrt(t1*t1+2*t1*t2*np.cos(k)+t2*t2))*interval1

E3 = 0*k

plt.figure(figsize=(8,7))
plt.plot(k, E1, color="black", alpha=0.6)
plt.plot(k, E2, color="black", alpha=0.6)
plt.plot(k, E3, color="steelblue", alpha=0.6)
my_x_ticks = np.arange(-np.pi, np.pi+0.1, np.pi/4)
my_y_ticks = np.arange(-np.pi, np.pi+0.1, np.pi/4)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel("quasimomentum", fontdict={'size': 16})
plt.ylabel("quasienergy", fontdict={'size':16})
plt.title('PBC-band-'+"a="+str(round(alpha,2))+"-b="+str(round(beta,2)), fontdict={'size': 20})
plt.savefig('PBC-band-'+"a="+str(round(alpha,2))+"-b="+str(round(beta,2))+'.jpg', dpi=300)
plt.show()