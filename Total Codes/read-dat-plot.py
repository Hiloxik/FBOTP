import matplotlib.pyplot as plt

file_position = 'D://大二下//科研//Topological Material//Floquet insulator//2D-HDP//codes'       # 文件位置
file_name = 'test_to_dat.dat'                       # 文件名
data_name = file_position+file_name         # 为了更改起来更方便

data0 = open(file_name,'r')
print(data0)
data1 = []
for line in data0:
    data1.append(float(line))
data0.close()
data2 = data1                    # 实际数据太长，截了其中的一小段
print(data2)

plt.plot(data2)                             # 绘图函数
plt.show()
