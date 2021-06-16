# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 6:04 下午
# @Author  : Michael Zhouy
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame()

# 单变量分布的柱状图
sns.distplot(df['y'])

# 两个变量之间的散点图
df.plot.scatter(x='col', y='y')

# 箱型图
sns.boxplot(x='col', y='y', data=df)


def corr_plot(df, k=10):
    """

    :param df:
    :param k: 找到相关系数前10的列名
    :return:
    """
    # 协方差矩阵的热力图
    corrmat = df.corr()  # DataFrame
    # sns.heatmap(corrmat, vmax=0.8, square=True)
    # corrmat.nlargest(k, 'SalePrice')  # 取与'y'相关系数最大的10行
    # corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': 10},
        yticklabels=cols.values,
        xticklabels=cols.values
    )
    plt.show()


def box_plot(df, columns, frows, fcols, figsize=(80, 60)):
    """
    箱型图
    :param df:
    :param columns:
    :param frows:
    :param fcols:
    :param figsize:
    :return:
    """
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        plt.subplot(frows, fcols, i)
        sns.boxplot(df[f], orient='v', width=0.5)
        plt.ylabel(f, fontsize=36)
    plt.show()


def dist_plot(df, columns, frows, fcols, figsize=(80, 60)):
    """
    直方图
    :param df:
    :param columns:
    :param frows:
    :param fcols:
    :param figsize:
    :return:
    """
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(df[f], fit=stats.norm)

        i += 1
        plt.subplot(frows, fcols, i)
        stats.probplot(df[f], plot=plt)
    plt.tight_layout()
    plt.show()


def kde_plot(train, test, columns, frows, fcols, figsize=(80, 60)):
    plt.figure(figsize=figsize)
    i = 0
    for f in columns:
        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.kdeplot(train[f], color='Red', shade=True)
        sns.kdeplot(test[f], color='Blue', shade=True)
        ax.set_xlabel(f)
        ax.set_ylabel('Frequency')
        ax.legend(['Train', 'Test'])
        i += 1
    plt.show()


def reg_plot(df, columns, frows, fcols, y='target', figsize=(80, 60)):
    plt.figure(figsize=figsize)
    i = 1
    for f in columns:
        ax = plt.subplot(frows, fcols, i)
        sns.regplot(x=f, y=y, data=df, ax=ax, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3}, line_kws={'color': 'k'})
        plt.xlabel(f)
        plt.ylabel(y)

        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.distplot(df[f].dropna())
        plt.xlabel(f)
    plt.show()
