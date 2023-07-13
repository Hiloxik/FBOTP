import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 


x=np.arange(0,2*np.pi+0.1,1)
interval1 = [1 if (i<=np.pi) else 0 for i in x]
interval2 = [1 if (i<=2*np.pi) else 0 for i in x]
interval3 = [1 if (i<=3*np.pi) else 0 for i in x]

y=x
z1=(np.pi-x)*interval1
z2=(2*np.pi-x)*interval2
z3=(3*np.pi-x)*interval3


plt.fill_between(x,y,z1,where=z1>=y,facecolor='green',interpolate=True)

plt.fill_between(x, y, color="gray", alpha=0.2)
plt.fill_between(x, z1, color="brown", alpha=0.2)
plt.plot(x, y, color="black", alpha=0.6)
plt.plot(x, z1, color="red", alpha=0.6)
plt.plot(x, z2, color="red", alpha=0.6)
plt.plot(x, z3, color="red", alpha=0.6)
plt.xlabel("$wt_2/\hbar$", fontdict={'size':16})
plt.ylabel("$vt_1/\hbar$", fontdict={'size':16})
my_x_ticks = np.arange(0, 2*np.pi+0.1, np.pi/4)
my_y_ticks = np.arange(0, 2*np.pi+0.1, np.pi/4)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.text(1.5, 0.5, 'C=1', fontdict={'size':16})
plt.text(2.5, 1.5, 'C=0', fontdict={'size':16})
plt.text(1.5, 2.5, 'C=-1', fontdict={'size':16})
plt.text(0.5, 1.5, 'C=0', fontdict={'size':16})
plt.title("phase diagram", fontdict={'size':20})
plt.savefig('phase diagram'+'.jpg', dpi=300)
plt.show()

