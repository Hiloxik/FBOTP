import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math  
import cmath
import seaborn as sns
import scipy
import functools
from scipy.linalg import expm

T = 2
J1 = ((3*np.pi/4)/(np.sqrt(2)))*(2/T)
J2 = ((np.pi/2)/(np.sqrt(2)))*(2/T)
J3 = 0
N = 20

cm = plt.cm.get_cmap('plasma') #get colorbar

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
        hopping_x1[i,i+1] = J1
        hopping_x1[i+1,i] = J1
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
        hopping_y1[i,i+1] = J1
        hopping_y1[i+1,i] = J1
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
    prob = np.dot(np.conjugate(eigenvector).T,eigenvector)
    for i in range(2*N*2*N):
        for j in range(2*N*2*N):
            if i != j:
                prob[i,j] = 0
            if i == j:
                prob[i,j] = prob[i,j]
    conv1 = np.zeros((2*N,2*N*2*N))
    for i in range(2*N):
        for j in range(2*N):
            conv1[i,i*2*N+j] = 1
    # print(conv1)
    conv2 = np.zeros((2*N*2*N,2*N))
    for i in range(2*N):
        for j in range(2*N):
            conv2[i*2*N+j,j] = 1
    # print(conv2)
    PROB = np.dot(np.dot(conv1,prob),conv2)
    for i in range(2*N):
        for j in range(2*N):
            PROB[i,j] = float(np.abs(PROB[i,j].real))
    sns.heatmap(np.abs(PROB))
    plt.show()

if __name__ == '__main__':
    main()