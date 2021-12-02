# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:51 下午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc

df = pd.DataFrame()

# 时间转换
date = '2021-12-01'
last_months = datetime.strptime(date, '%Y-%m-%d') + relativedelta(months=-1)
last_months = datetime.strftime(last_months, '%Y-%m-%d')
print(last_months)

# rolling时间窗口
df.groupby('id_road')['TTI'].rolling('60min', closed='left', min_periods=6).mean()
df.sort_values(by='date', inplace=True)
# using center=false to assign values on window's last row
df['val_rolling_7_mean'] = df.groupby('group_key')['val'].transform(lambda x: x.rolling(7, center=False).mean())


def group_rolling(df, num_cols):
    """
    滑窗
    :param df:
    :param num_cols:
    :return:
    """
    for i in num_cols:
        for j in ['90min', '120min']:
            df.set_index('datetime', inplace=True)
            tmp = df[i].rolling(j, closed='left', min_periods=1).agg({
                '{}_{}_rolling_mean'.format(i, j): 'mean',
                '{}_{}_rolling_median'.format(i, j): 'median',
                '{}_{}_rolling_max'.format(i, j): 'max',
                '{}_{}_rolling_min'.format(i, j): 'min',
                '{}_{}_rolling_sum'.format(i, j): 'sum',
                '{}_{}_rolling_std'.format(i, j): 'std',
                '{}_{}_rolling_skew'.format(i, j): 'skew'
            })
            tmp.reset_index(inplace=True)
            df.reset_index(inplace=True)
            df = df.merge(tmp, on=['datetime'], how='left')
            del tmp
            gc.collect()
    return df


def get_time_feature(df, col, keep=False):
    """
    为df增加时间特征列, 包括:年,月,日,小时,dayofweek,weekofyear
    :param df:
    :param col: 时间列的列名
    :param keep: 是否保留原始时间列
    :return:
    """
    df_copy = df.copy()
    prefix = col + "_"

    df_copy[col] = pd.to_datetime(df_copy[col])
    df_copy[prefix + 'year'] = df_copy[col].dt.year
    df_copy[prefix + 'month'] = df_copy[col].dt.month
    df_copy[prefix + 'day'] = df_copy[col].dt.day
    df_copy[prefix + 'hour'] = df_copy[col].dt.hour
    df_copy[prefix + 'weekofyear'] = df_copy[col].dt.weekofyear
    df_copy[prefix + 'dayofweek'] = df_copy[col].dt.dayofweek
    df_copy[prefix + 'is_wknd'] = df_copy[col].dt.dayofweek // 4
    df_copy[prefix + 'quarter'] = df_copy[col].dt.quarter
    df_copy[prefix + 'is_month_start'] = df_copy[col].dt.is_month_start.astype(int)
    df_copy[prefix + 'is_month_end'] = df_copy[col].dt.is_month_end.astype(int)
    if keep:
        return df_copy
    else:
        return df_copy.drop([col], axis=1)


df = get_time_feature(df, "time_col")

# lag特征, 表示同一元素历史时间点的值(例如在这个赛题中，同一个unit昨天、前天、上周对应的使用量)
keys = ['unit']
val = 'qty'
lag = 1
df.groupby(keys)[val].transform(lambda x: x.shift(lag))

# 滑动窗口统计特征：历史时间窗口内的统计值
keys = ['unit']
val = 'qty'
window = 7
df.groupby(keys)[val].transform(
  lambda x: x.rolling(window=window, min_periods=3, win_type="triang").mean())
df.groupby(keys)[val].transform(
  lambda x: x.rolling(window=window, min_periods=3).std())

# 指数加权移动平均
keys = ['unit']
val = 'qty'
lag = 1
alpha=0.95
df.groupby(keys)[val].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
