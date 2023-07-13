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

J1 = 3
J2 = 5
J = np.linspace(1000,1020,20)
L =len(J)

def hamiltonian(dk=0.001):
    dk = dk # 变化率
    k = np.arange(-np.pi, np.pi, dk)
    C = [0 for index in range(L)]
    for j in range(L):
        J3 = J[j]
        x = (J1+J2*np.cos(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
        y = (J2*np.sin(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
        z = (2*J3*np.sin(k))/(np.sqrt(math.pow(J1,2)+math.pow(J2,2)+2*J1*J2*np.cos(k)+4*math.pow(J3,2)*(np.sin(k)**2)))
        area_list = [] # 存储每一微小步长的曲线长度
        for i in range(1,len(k)):
        # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
            dl_i = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2) 
            # 将计算结果存储起来
            area_list.append(dl_i)
            area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度
        C[j] = area
        # print("三维空间曲线长度：{:.4f}".format(area))
    plt.plot(J,C)
    plt.show()

if __name__ == '__main__':
    hamiltonian()