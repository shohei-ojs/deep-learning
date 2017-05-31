# coding: UTF-8
import numpy as np

# ソフトマックス関数
def softmax(a):
    # 一番大きい値を取得
    c = np.max(a)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a

    return y


a = [23.0, 0.94, 5.46]
print (softmax(a))
# [  9.99999976e-01   2.62702205e-10   2.41254141e-08]