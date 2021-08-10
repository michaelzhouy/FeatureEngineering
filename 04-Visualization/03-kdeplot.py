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
