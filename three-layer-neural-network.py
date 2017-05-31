# coding: UTF-8
import numpy as np

# シグモイド関数(活性化関数)
# 値を0.0~1.0の間に標準化
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 重みとバイアスの初期化
def init_network():
    network = {}
    # 入力層から第一層への重みとバイアス
    network['W1'] =np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] =np.array([0.1,0.2,0.3])
    # 第一層から第二層への重みとバイアス
    network['W2'] =np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] =np.array([0.1,0.2])
    # 第二層から第三層への重みとバイアス
    network['W3'] =np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] =np.array([0.1,0.2])

    return network
    

def forward( network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    # テンソルxとW1の積を求めバイアスb1をたす
    a1 = np.dot(x,W1) + b1
    # テンソルa1に活性化関数sigmoid()を適用
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = a3

    return y

network = init_network()
x = np.array([0.1,0.5])
y = forward(network, x)
print(y)