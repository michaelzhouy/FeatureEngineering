# -*- coding: utf-8 -*-
# @Time    : 2021/7/19 11:35 上午
# @Author  : Michael Zhouy


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


arange = np.linspace(0, 20, 200)
normal_data = np.sin(arange)
nosie = normal_data.copy()
num_nosie = 40
index = np.random.randint(0, 200, (num_nosie, ))
for i in range(num_nosie):
    nosie[index[i]] = nosie[index[i]] + np.random.normal(-0.08, 0.08)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(arange, normal_data)
axes[1].plot(arange, nosie)
plt.show()


def mean_filter(kernel_size, data):
    # 均值滤波
    if kernel_size % 2 == 0 or kernel_size <= 1:
        print('kernel_size滤波核的需为大于1的奇数')
        return
    else:
        padding_data = []
        mid = kernel_size // 2
        for i in range(mid):
            padding_data.append(0)
        padding_data.extend(data.tolist())
        for i in range(mid):
            padding_data.append(0)
    result = []
    for i in range(0, len(padding_data) - 2 * mid, 1):
        temp = 0
        for j in range(kernel_size):
            temp += padding_data[i + j]
        temp = temp / kernel_size
        result.append(temp)
    return result


nosie_mean_filter = mean_filter(3, nosie)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(arange, normal_data)
axes[1].plot(arange, nosie)
axes[2].plot(arange, nosie_mean_filter)
plt.show()