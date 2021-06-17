# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:51 下午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
import gc

df = pd.DataFrame()

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
