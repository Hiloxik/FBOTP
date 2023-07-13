import matplotlib.pyplot as plt
import numpy as np

phi = 0.2
m = 20

filename1 = 'D:\\科研\\Topomat\\Floquet insulator\\3D-FBOTP\\files\\DP795-zx-1.txt'
filename2 = 'D:\\科研\\Topomat\\Floquet insulator\\3D-FBOTP\\files\\DP795-zx-2.txt'

X = []
with open(filename1, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [float(s) for s in line.split()]#4
        X.append(value[0])#5

Y = []
with open(filename2, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [float(s) for s in line.split()]#4
        Y.append(value[0])#5
C = np.linspace(0,m,len(X))

plt.scatter(C,X)
plt.scatter(C,Y)
plt.show()
