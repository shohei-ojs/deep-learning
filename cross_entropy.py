# coding: UTF=8
import numpy as np

# 交差エントロピー誤差(E = -Σt)
def cross_entropy_error(y,t):
    delta = 10e-7
    # np.log(0)にならないよう小さな値をyにたす
    return -np.sum(t * np.log(y + delta))



y1 = 0.9
y2 = 0.1
t = 1
print(cross_entropy_error(y,t))
#y1:2.30257509304
#y2:0.105359404547