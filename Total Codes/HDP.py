import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import math
import functools

#parameters
t1 = 0.3*np.pi
t2 = 3.5*np.pi
print(t1,t2)
T = t1+t2
n = 3
N = 1000
k_array = np.linspace(-np.pi,np.pi,N)
t_array = np.linspace(0,T,30)
gap = 0
l = len(k_array)
L = len(t_array)

#pauli matrix
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

#generalized logarithm
def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j*argument

#anomalous periodic operator
def U(k,t):
    UF = np.dot(np.cos(t2)*s0-1j*np.sin(t2)*np.cos(n*k)*s1-1j*np.sin(t2)*np.sin(n*k)*s2,np.cos(t1)*s0-1j*np.sin(t1)*s1)
    a0 = np.cos(t1)*np.cos(t2)-np.sin(t1)*np.sin(t2)*np.cos(n*k)
    a1 = -np.cos(t1)*np.sin(t2)*np.cos(n*k)-np.sin(t1)*np.cos(t2)
    a2 = -np.cos(t1)*np.sin(t2)*np.sin(n*k)
    a3 = np.sin(t1)*np.sin(t2)*np.sin(n*k)
    f = np.arccos(a0)
    theta = np.arccos(a3/np.sqrt(a1**2+a2**2+a3**2))
    phi = (a1+1j*a2)/(np.sqrt(a1**2+a2**2))
    pk = np.matrix([[np.cos(theta/2)**2,np.sin(theta/2)*np.cos(theta/2)*np.conjugate(phi)],[np.sin(theta/2)*np.cos(theta/2)*phi,np.sin(theta/2)**2]])
    mk = np.matrix([[np.sin(theta/2)**2,-np.sin(theta/2)*np.cos(theta/2)*np.conjugate(phi)],[-np.sin(theta/2)*np.cos(theta/2)*phi,np.cos(theta/2)**2]])

    r = int(t//T)

    if 0 <= t%T < t1:
        U1 = np.dot(np.cos(t%T)*s0-1j*np.sin(t%T)*s1,np.linalg.matrix_power(UF,r))
    if t1 <= t%T < T:
        U1 = np.dot(np.cos(t%T-t1)*s0-1j*np.sin(t%T-t1)*np.cos(n*k)*s1-1j*np.sin(t%T-t1)*np.sin(n*k)*s2,np.dot(np.cos(t1)*s0-1j*np.sin(t1)*s1,np.linalg.matrix_power(UF,r)))

    if gap == 0:
        U2 = np.exp(-1j*f*((t)/T))*pk+np.exp(-1j*(2*np.pi-f)*((t)/T))*mk
    if gap == np.pi:
        U2 = np.exp(-1j*f*((t)/T))*pk+np.exp(1j*f*((t)/T))*mk
    return np.dot(U1,U2)

#polarization calculation
def main():
    nu_array = [0 for index in range(L)]
    un_array = [0 for index in range(L)]
    for j in range(L):
        t = t_array[j]
        W = s0
        for k in k_array:
            UU0 = U(k,t)
            # print(U(k,t)-U(k,t+2*T))
            UU1 = np.conjugate(U(k+(2*np.pi/N),t)).T
            Q = (1-1/(2*(3*n)))*s0+(1/(2*(3*n)))*np.dot(UU1,UU0)
            u,s,v = np.linalg.svd(Q) #SVD
            QQ = np.dot(u,v)
            W = np.dot(QQ,W) 
        print(np.linalg.det(W),j)
        eigenvalue, eigenvector = np.linalg.eig(W)
        nu_array[j] = (1j/(2*np.pi))*llog(eigenvalue[0],0)
        un_array[j] = (1j/(2*np.pi))*llog(eigenvalue[1],0)
    tt = np.linspace(0,T,100)
    z1 = 0*tt+0.5
    plt.scatter(t_array,nu_array)
    plt.scatter(t_array,un_array)
    plt.plot(tt,z1,c='black')
    plt.xlabel("time", fontdict={'size': 16})
    plt.ylabel("polarization", fontdict={'size':16})
    plt.title("dynamical polarization", fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()