import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


lx_list = np.linspace(-4*np.pi,4*np.pi,100)
ly_list = np.linspace(4*np.pi,-4*np.pi,100)
gx_list = np.linspace(0,np.pi,100)
gy_list = np.linspace(0,np.pi,100)

def f(a,b,c,d):
    f = np.cos(np.sqrt(a**2+b**2))*np.cos(np.sqrt(c**2+d**2))+np.sin(np.sqrt(a**2+b**2))*np.sin(np.sqrt(c**2+d**2))*((a*c+b*d)/(np.sqrt(a**2+b**2)*np.sqrt(c**2+d**2)))
    return f

def main():
    # phase = np.zeros((100,100))
    f = []
    gx = np.pi/4
    for i in range(len(lx_list)):
        lx = lx_list[i]
        for j in range(len(ly_list)):
            ly = ly_list[j]
            for k in range(len(gy_list)):
                gy = gy_list[k]
                f.append(f(lx,ly,gx,gy))
    
    
    plt.show()

if __name__ == '__main__':
    main()