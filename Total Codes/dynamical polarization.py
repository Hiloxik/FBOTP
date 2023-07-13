import numpy as np
import matplotlib.pyplot as plt 
import cmath 
import functools

J1 = 1
J2 = 2
J3 = 5
N = 10000
k_array = np.linspace(-np.pi,np.pi,N)
if J3 == 0:
    T = (np.pi)/(2*(J1+J2))
else:
    T = (np.pi)/(2*np.sqrt(J1**2+J2**2+(((J1**2)*(J2**2))/(4*J3**2))+4*J3**2))
t_array = np.linspace(0,T,50)
l = len(k_array)
L = len(t_array)

s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

def llog(z, theta):
    modulus = np.abs(z)
    argument = np.angle(z)
    if theta-2*np.pi <= argument < theta:
        argument = argument
    else:
        argument = theta-2*np.pi+np.mod(argument-theta, 2*np.pi)
    return np.log(modulus) + 1j * argument

def U(k,t):
    f = np.sqrt(J1**2+J2**2+2*J1*J2*np.cos(k)+4*(J3**2)*np.sin(k)**2)*T
    theta = np.arccos((2*J3*np.sin(k))/(np.sqrt(J1**2+J2**2+2*J1*J2*np.cos(k)+4*(J3**2)*np.sin(k)**2)))
    phi = (J1+J2*np.exp(1j*k))/(np.sqrt(J1**2+J2**2+2*J1*J2*np.cos(k)))
    U1 = np.cos(f*(t/T))*s0-1j*np.sin(f*(t/T))*np.sin(theta)*(phi.real)*s1-1j*np.sin(f*(t/T))*np.sin(theta)*(phi.imag)*s2-1j*np.sin(f*(t/T))*np.cos(theta)*s3
    U2 = np.exp(-1j*np.pi*(t/T))*(np.cos((-f-np.pi)*(t/T))*s0-1j*np.sin((-f-np.pi)*(t/T))*np.sin(theta)*(phi.real)*s1-1j*np.sin((-f-np.pi)*(t/T))*np.sin(theta)*(phi.imag)*s2-1j*np.sin((-f-np.pi)*(t/T))*np.cos(theta)*s3)
    return np.dot(U1,U2)

def main():
    nu_array = [0 for index in range(L)]
    un_array = [0 for index in range(L)]
    for j in range(L):
        t = t_array[j]
        W = s0
        for k in k_array:
            UU0 = U(k,t)
            print(U(k,t)-U(k,t-T))
            UU1 = np.conjugate(U(k+(2*np.pi/N),t)).T
            Q = (1/(2))*s0+(1-1/(2))*np.dot(UU1,UU0)
            W = np.dot(Q,W) 
        # print(np.linalg.det(W))
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
    plt.title("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3), fontdict={'size': 20})
    # plt.savefig("dynamical polarization-"+str(N)+'cells-'+'J1/J2='+str(J1/J2)+'-J3='+str(J3)+'.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()