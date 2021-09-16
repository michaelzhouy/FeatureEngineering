# -*- coding: utf-8 -*-
# @Time    : 2021/8/3 9:52 上午
# @Author  : Michael Zhouy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(df1, df2, df3, x):
    fig = plt.figure()
    ax = plt.axes()
    sns.kdeplot(data=df1, x=x)
    sns.kdeplot(data=df2, x=x)
    sns.kdeplot(data=df3, x=x)
    plt.legend(['origin', 'sample', 'new'])
    plt.show()


def multi_plot(df1, df2, feats):
    rows, cols = 5, 4
    plt.figure(figsize=(4 * rows, 4 * cols))
    i = 0
    for col in feats:
        i += 1
        ax = plt.subplot(rows, cols, i)
        sns.kdeplot(df1[col], ax=ax)
        sns.kdeplot(df2[col], ax=ax)
    plt.show()
