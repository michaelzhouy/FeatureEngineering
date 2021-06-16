# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 6:23 下午
# @Author  : Michael Zhouy
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def identify_single_unique(df):
    """
    单一值
    :param df:
    :return:
    """
    unique_cnts = df.nunique()
    unique_cnts = unique_cnts.sort_values(by='nunique', ascending=True)
    to_drop = unique_cnts[unique_cnts == 1].index.to_list()
    print('{} features with a single unique value.\n'.format(len(to_drop)))
    return to_drop


def identify_missing(df, missing_threshold):
    """
    缺失率
    :param df:
    :param missing_threshold:
    :return:
    """
    missing_rate = df.isnull().sum() / len(df)
    missing_rate = missing_rate.sort_values(ascending=False)
    print(missing_rate)
    to_drop = missing_rate[missing_rate > missing_threshold].index.to_list()
    print('{} features with greater than {} missing values.\n'.format(len(to_drop), missing_threshold))
    return to_drop


def overfit_reducer(df, threshold=99.9):
    """
    计算每列中取值的分布，返回单一值占比达到阈值的列名
    :param df:
    :param threshold:
    :return:
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > threshold:
            overfit.append(i)
    return overfit


def missing_percentage(df):
    """
    计算每列的缺失率
    @param df:
    @return:
    """
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)[round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def timestamp2string(timeStamp):
    """
    将时间戳转换为str对象
    :param timeStamp:
    :return:
    """
    try:
        d = datetime.datetime.fromtimestamp(timeStamp)
        # str类型
        str = d.strftime('%Y-%m-%d %H:%M:%S.%f')
        return str
    except Exception as e:
        print(e)
        return ''


def get_datetime(df, time_col, type='hour'):
    """
    获取年、月、日、小时等
    :param df:
    :param time_col:
    :param type: 'hour' 'day' 'month' 'year' 'weekday'
    :return:
    """
    if type == 'hour':
        df['hour'] = df[time_col].map(lambda x: int(str(x)[11: 13]))
    elif type == 'day':
        df['day'] = df[time_col].map(lambda x: int(str(x)[8: 10]))
    elif type == 'month':
        df['month'] = df[time_col].map(lambda x: int(str(x)[5: 7]))
    elif type == 'year':
        df['year'] = df[time_col].map(lambda x: int(str(x)[0: 4]))
    elif type == 'weekday':
        df['weekday'] = pd.to_datetime(df[time_col]).map(lambda x: x.weekday())
    return df


def null_analysis(df, p='f24'):
    """
    特征的缺失情况随着时间推移的变化
    :param df:
    :param p: 某一特征
    :return:
    """
    a = pd.DataFrame(df.groupby('ndays')[p].apply(lambda x: sum(pd.isnull(x))) / df.groupby('ndays')['ndays'].count()).reset_index()
    a.columns = ['ndays', p]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'], a[p])
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('miss_rate_' + p)
    plt.title('miss_rate_' + p)


def value_analysis(df, p='f24'):
    """
    特征取值随着时间推移的变化
    :param df:
    :param p: 某一特征
    :return:
    """
    a = pd.DataFrame(df.groupby('ndays')[p].mean()).reset_index()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'], a[p])
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('mean_of_' + p)
    plt.title('distribution of ' + p)
