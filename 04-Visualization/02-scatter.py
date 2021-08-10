# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 9:52 上午
# @Author  : Michael Zhouy
import matplotlib.pyplot as plt


def plot(df1, df2, df3, df4):
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(df1['x'].astype(int), df1['y'].astype(int), c='r', s=2, marker='o', label='train')
    plt.scatter(df2['x'].astype(int), df2['y'].astype(int), c='g', s=5, marker='o', label='valid')
    plt.scatter(df3['x'].astype(int), df3['y'].astype(int), c='b', s=8, marker='x', label='test')
    plt.scatter(df4['x'].astype(int), df4['y'].astype(int), c='k', s=2, alpha=0.1, label='new test')
    plt.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.show()
