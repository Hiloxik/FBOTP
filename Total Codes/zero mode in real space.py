import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools

cm = plt.cm.get_cmap('plasma')

J1 = 2 #intercell hopping
ratio = np.linspace(0,4,50)
J2 = J1*ratio
l = len(J2)
N=5 #number of unit cells
width=2*N
length=2*N

def H(J):   # 方格子哈密顿量
    hopping_x = np.zeros((length, length))
    hopping_y = np.zeros((width, width))
    hopping_xy = np.zeros((width, width))
    for i in range(0,length-1,2):
        hopping_x[i, i+1] = J1
        hopping_x[i+1, i] = J1
    for i in range(1,length-1,2):
        hopping_x[i, i+1] = J
        hopping_x[i+1, i] = J
    for i in range(0,width-1,2):
        hopping_y[i, i+1] = J1
        hopping_y[i+1, i] = J1
    for i in range(1,width-1,2):
        hopping_y[i, i+1] = J
        hopping_y[i+1, i] = J
    for i in range(0,length,2):
        hopping_xy[i, i] = -1
    for i in range(1,length,2):
        hopping_xy[i, i] = 1
    h = np.kron(hopping_x, np.eye(width))+np.kron(hopping_xy, hopping_y)
    return h

def main():
    for j in range((2*N)*(2*N)):
        zero = [0 for index in range(l)]
        for i in range(l):
            J = J2[i]
            eigenvalue, eigenvector = np.linalg.eig(H(J))
            eigenvalue.sort()
            zero[i] = eigenvalue[j].real
            # print(zero[i])
        plt.plot(ratio, zero)
    # y = 0*J
    # plt.plot(J,y,c='black')
    plt.xlabel("J2/J1", fontdict={'size': 16})
    plt.ylabel("energy", fontdict={'size':16})
    # plt.title('OBC-'+str(N)+"cells-"+"J1="+str(J)+"-"+"J2="+str(J3), fontdict={'size': 20})
    # plt.savefig('OBC-'+str(N)+"cells-"+"J1="+str(J1)+"-"+"J2="+str(J2)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()