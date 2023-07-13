import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn 
import scipy
import functools
import time

cm = plt.cm.get_cmap('plasma')


J3 = np.linspace(0,500,100)
l = len(J3)

def E(k,J,J1,J2):
    E = np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J,2)*math.pow(np.sin(k),2))
    return E

def Ep(k,J1,J2):
    Ep = np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k))
    return Ep

def main():
    delta_1 = 1e-10
    delta_2 = 1e-3 #integration step
    W1 = [0 for index in range(l)]
    for i in range(0,l,1):
        J1 = 1
        J2 = 2
        J = J3[i]
        C1=0
        for k in np.arange(-np.pi, np.pi, delta_2):
            PH0 = cmath.log((J1+J2*np.exp(1j*k))/Ep(k,J1,J2))
            PH1 = cmath.log((J1+J2*np.exp(1j*(k+delta_1)))/Ep(k+delta_1,J1,J2))
            C1 = C1+(1-(2*J*np.sin(k))/(E(k,J,J1,J2)))*((PH1-PH0)/(delta_1))*delta_2
        W1[i] = round(((1j*C1)/(2*np.pi)).real,3)
    W2 = [0 for index in range(l)]
    for i in range(0,l,1):
        J1 = 2
        J2 = 1
        J = J3[i]
        C2=0
        for k in np.arange(-np.pi, np.pi, delta_2):
            PH0 = cmath.log((J1+J2*np.exp(1j*k))/Ep(k,J1,J2))
            PH1 = cmath.log((J1+J2*np.exp(1j*(k+delta_1)))/Ep(k+delta_1,J1,J2))
            C2 = C2+(1-(2*J*np.sin(k))/(E(k,J,J1,J2)))*((PH1-PH0)/(delta_1))*delta_2
        W2[i] = round(((1j*C2)/(2*np.pi)).real,3)
    plt.scatter(J3,W1,s=10,c=W1,cmap=cm)
    plt.scatter(J3,W2,s=10,c=W2,cmap=cm)
    plt.colorbar()
    plt.xlabel("J3", fontdict={'size': 16})
    plt.ylabel("winding number", fontdict={'size':16})
    # my_y_ticks = np.arange(-0.1, 1.1, 0.1)
    # plt.yticks(my_y_ticks)
    plt.savefig('winding number-'+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()