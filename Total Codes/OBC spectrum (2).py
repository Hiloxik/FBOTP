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
J11 = 1.75*(2/T)
J12 = ((np.pi)/(np.sqrt(2)))*(2/T)
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
    plt.scatter(k, quasienergy, c='red', s=30)
    plt.plot(x, y0, color="black", alpha=0.6)
    plt.plot(x, y1, color="black", alpha=0.6)
    plt.plot(x, y2, color="black", alpha=0.6)
    plt.xlabel("eigenstate", fontdict={'size': 16})
    plt.ylabel("quasienergy", fontdict={'size':16})
    plt.yticks([-np.pi/T,0,np.pi/T],[r'$-\pi$','0',r'$\pi$'],size=18,weight='bold')
    # plt.title("OBC-"+str(N)+"cells-"+"a="+str(round(aalpha,2))+"-b="+str(round(bbeta,2)), fontdict={'size': 20})
    # plt.savefig('OBC-'+str(N)+"-"+str(round(aalpha,2))+"-"+str(round(bbeta,2))+'.jpg', dpi=300)
    plt.show()

    #wavefunction distribution
    X, Y = np.meshgrid(np.linspace(0,2*N,2*N), np.linspace(0,2*N,2*N)) 
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x', fontsize=15, color='black',fontweight='bold')  
    ax.set_ylabel('y', fontsize=15, color='black',fontweight='bold')  
    ax.set_zlabel('probability', fontsize=15, color='black',fontweight='bold')
    edgewave = eigenvector[:,(quasienergy>-np.pi/T-0.05)&((quasienergy<-np.pi/T+0.05))]
    for i in range(edgewave.shape[1]):
            Plot0 = edgewave[:,i]
            NumToPlot0 = np.abs(Plot0)**2
            p0 = np.reshape(NumToPlot0,(2*N,2*N)).T
            ax.plot_surface(X, Y, p0, cmap='plasma')
            ax.contour(X, Y, p0, 1000, zdir = 'z', offset = 0.25, cmap = plt.get_cmap('rainbow'))
    plt.show()

if __name__ == '__main__':
    main()