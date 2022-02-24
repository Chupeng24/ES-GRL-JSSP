import numpy as np
import os.path as path

# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4], [5, 6]])

d = path.dirname(__file__)
parent_path = path.dirname(d) #获得d所在的目录,即d的父级目录

# 保存.npy文件
np.save(parent_path+"./DataGen/arr.npy", arr)

# 读取.npy文件
np.load("./arr.npy")
print(arr)