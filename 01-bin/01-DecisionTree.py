# -*- coding:utf-8 -*-
# Time   : 2021/4/3 22:50
# Email  : 15602409303@163.com
# Author : Zhou Yang
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def optimalBinningBoundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    """
    optimalBinningBoundary(x=data['RevolvingUtilizationOfUnsecuredLines'], y=data['SeriousDlqin2yrs'])
    :param x:
    :param y:
    :param nan:
    :return:
    """
    boundary = []  # 待return的分箱边界值列表
    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_leaf_nodes=6,  # 最大叶子节点数
        min_samples_leaf=0.05  # 叶子节点样本数量最小占比
    )

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()
    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    return boundary


def featureWoeIv(x: pd.Series, y: pd.Series, nan: float = -999.) -> pd.DataFrame:
    """
    计算变量各个分箱的WOE、IV值，返回一个DataFrame
    :param x:
    :param y:
    :param nan:
    :return:
    """
    x = x.fillna(nan)
    boundary = optimalBinningBoundary(x, y, nan)  # 获得最优分箱边界值列表
    df = pd.concat([x, y], axis=1)  # 合并x、y为一个DataFrame，方便后续计算
    df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名
    df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间

    grouped = df.groupby('bins')['y']  # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()  # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()  # 总客户占比

    result_df['bad_rate'] = result_df['bad'] / result_df['total']  # 坏比率

    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])  # WOE
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV

    print("该变量IV = {}".format(result_df['iv'].sum()))

    return result_df
