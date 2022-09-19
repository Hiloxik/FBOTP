import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn as sns
import scipy
import functools
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D 

T = 2
J11 = (1.75)*(2/T)
J12 = ((4*np.pi/4)/(np.sqrt(2)))*(2/T)
J2 = ((np.pi/4)/(np.sqrt(2)))*(2/T)
J3 = 0
N = 20
d = 2*N*2*N

cm = plt.cm.get_cmap('plasma') #get colorbar

#states-sorting function
def ssort(x,y):
    e = -1j*np.log(x)/T
    x_sort_idx = np.argsort(e)[::1]
    e = np.sort(e)[::1]
    y = y[:,x_sort_idx]
    x = e
    return x,y
    
def fix(x,d):
    Phase = np.zeros(shape=(d,d),dtype=complex)
    for j in range(d):
        Phase[j,j] = np.exp(-1j*cmath.phase(x[0,j]))
    x = np.dot(x,Phase)
    return x

#calculate the evolution operator
def Floquet():
    hopping_x1 = np.zeros((2*N,2*N))
    hopping_y1 = np.zeros((2*N,2*N))
    hopping_x2 = np.zeros((2*N,2*N))
    hopping_y2 = np.zeros((2*N,2*N))
    hopping_xy = np.zeros((2*N,2*N))
    hopping_yx = np.zeros((2*N,2*N))
    hopping_x3 = np.zeros((2*N,2*N),dtype=complex)

    for i in range(0,2*N-1,2):
        hopping_x1[i,i+1] = J11
        hopping_x1[i+1,i] = J11
    for i in range(1,2*N-1,2):
        hopping_x1[i,i+1] = 0
        hopping_x1[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_x3[i,i+1] = 0
        hopping_x3[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_x3[i,i+1] = 0
        hopping_x3[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_y1[i,i+1] = J12
        hopping_y1[i+1,i] = J12
    for i in range(1,2*N-1,2):
        hopping_y2[i,i+1] = 0
        hopping_y2[i+1,i] = 0
    for i in range(0,2*N-1,2):
        hopping_xy[i, i] = -1
    for i in range(1,2*N,2):
        hopping_xy[i, i] = 1
    for i in range(0,2*N-1,2):
        hopping_yx[i, i] = 1
    for i in range(1,2*N,2):
        hopping_yx[i, i] = -1
    h1 = np.kron(hopping_x1, np.eye(2*N))+np.kron(hopping_xy, hopping_y1)+np.kron(hopping_x3, hopping_yx)

    for i in range(0,2*N-1,2):
        hopping_x2[i,i+1] = 0
        hopping_x2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_x2[i,i+1] = J2
        hopping_x2[i+1,i] = J2
    for i in range(0,2*N-1,2):
        hopping_y2[i,i+1] = 0
        hopping_y2[i+1,i] = 0
    for i in range(1,2*N-1,2):
        hopping_y2[i,i+1] = J2
        hopping_y2[i+1,i] = J2
    h2 = np.kron(hopping_x2, np.eye(2*N))+np.kron(hopping_xy, hopping_y2)
    UF = np.dot(expm(-1j*h1*T/4),np.dot(expm(-1j*h2*T/2),expm(-1j*h1*T/4)))
    return UF

def main():
    eigenvalue, eigenvector = np.linalg.eig(Floquet())
    # print(eigenvector)
    # eigenvector = fix(eigenvector,d)
    quasienergy, eigenvector = ssort(eigenvalue,eigenvector)
    #number the eigenvalues
    k = np.arange(0,2*N*2*N)
    z = quasienergy
    x=np.arange(0,2*N*2*N+0.1,0.1)
    y0=0*x
    y1=0*x+(np.pi)/(T)
    y2=0*x-(np.pi)/(T)

    #OBC spectrum
    plt.scatter(k, quasienergy, s=30, color='red')
    plt.plot(x, y0, color="steelblue", alpha=0.6)
    plt.plot(x, y1, color="steelblue", alpha=0.6)
    plt.plot(x, y2, color="steelblue", alpha=0.6)
    plt.xlabel("eigenstate", fontdict={'size': 16})
    plt.ylabel("quasienergy", fontdict={'size':16})
    my_y_ticks = np.arange(-(np.pi)/(T), (np.pi)/(T)+(np.pi)/(4*(T)), (np.pi)/(4*(T)))
    plt.yticks(my_y_ticks)
    # plt.title("OBC-"+str(N)+"cells-"+"a="+str(round(aalpha,2))+"-b="+str(round(bbeta,2)), fontdict={'size': 20})
    # plt.savefig('OBC-'+str(N)+"-"+str(round(aalpha,2))+"-"+str(round(bbeta,2))+'.jpg', dpi=300)
    plt.show()

    #wavefunction distribution
    edgewave = eigenvector[:,(quasienergy>0/T-0.05)&((quasienergy<0/T+0.05))]
    Plot0 = edgewave[:,math.ceil(edgewave.shape[1]/2)-2]
    NumToPlot0 = np.abs(Plot0)
    NumToPlot0 = np.reshape(NumToPlot0,(2*N,2*N))
    Plot1 = edgewave[:,math.ceil(edgewave.shape[1]/2)-1]
    NumToPlot1 = np.abs(Plot1)
    NumToPlot1 = np.reshape(NumToPlot1,(2*N,2*N))
    Plot2 = edgewave[:,math.ceil(edgewave.shape[1]/2)]
    NumToPlot2 = np.abs(Plot2)
    NumToPlot2 = np.reshape(NumToPlot2,(2*N,2*N))
    Plot3 = edgewave[:,math.ceil(edgewave.shape[1]/2)+1]
    NumToPlot3 = np.abs(Plot3)
    NumToPlot3 = np.reshape(NumToPlot3,(2*N,2*N))
    plt.subplot(221)
    sns.heatmap(NumToPlot0, cmap='afmhot', linewidths=0.5, linecolor='white')
    plt.subplot(222)
    sns.heatmap(NumToPlot1, cmap='afmhot', linewidths=0.5, linecolor='white')
    plt.subplot(223)
    sns.heatmap(NumToPlot2, cmap='afmhot', linewidths=0.5, linecolor='white')
    plt.subplot(224)
    sns.heatmap(NumToPlot3, cmap='afmhot', linewidths=0.5, linecolor='white')
    plt.show()

if __name__ == '__main__':
    main()