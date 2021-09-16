# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:14 下午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

df = pd.DataFrame()

# 分组排名
df.groupby('uid')['time'].rank('dense')

# 每天的通话次数
voc_day_cnt_res = df.groupby(['phone_no_m', 'voc_day'])['phone_no_m'].count().unstack()
for i in df['voc_day'].unique():
    df['voc_day{}_count'.format(i)] = df['phone_no_m'].map(voc_day_cnt_res[i])


def arithmetic(df, num_cols):
    """
    数值特征之间的加减乘除，x * x / y, log(x) / y
    :param df:
    :param num_cols: 交叉用的数值特征
    :return:
    """
    for i in tqdm(range(len(num_cols))):
        for j in range(i + 1, len(num_cols)):
            colname_add = '{}_{}_add'.format(num_cols[i], num_cols[j])
            colname_substract = '{}_{}_subtract'.format(num_cols[i], num_cols[j])
            colname_multiply = '{}_{}_multiply'.format(num_cols[i], num_cols[j])
            df[colname_add] = df[num_cols[i]] + df[num_cols[j]]
            df[colname_substract] = df[num_cols[i]] - df[num_cols[j]]
            df[colname_multiply] = df[num_cols[i]] * df[num_cols[j]]

    for f1 in tqdm(num_cols):
        for f2 in num_cols:
            if f1 != f2:
                colname_ratio = '{}_{}_ratio'.format(f1, f2)
                df[colname_ratio] = df[f1].values / (df[f2].values + 0.001)
    return df


def cat_cat_stats(df, id_col, cat_cols):
    """
    类别特征之间的groupby统计特征
    :param df:
    :param id_col:
    :param cat_cols:
    :return:
    """
    for f1 in cat_cols:
        for f2 in cat_cols:
            if f1 != f2:
                tmp = df.groupby([id_col, f1], as_index=False)[f2].agg({
                    '{}_{}_cnt'.format(f1, f2): 'count',
                    '{}_{}_nunique'.format(f1, f2): 'nunique',
                    '{}_{}_mode'.format(f1, f2): lambda x: x.value_counts().index[0],  # 众数
                    '{}_{}_mode_cnt'.format(f1, f2): lambda x: x.value_counts().values[0]  # 众数出现的次数
                })
                tmp['{}_{}_rate'.format(f1, f2)] = tmp['{}_{}_nunique'.format(f1, f2)] / tmp['{}_{}_cnt'.format(f1, f2)]
                df = df.merge(tmp, on=[id_col, f1], how='left')
                del tmp
                gc.collect()
    return df


def cat_num_stats(df, cat_cols, num_cols):
    """
    类别特征与数据特征groupby统计特征, 简单版
    :param df:
    :param cat_cols:
    :param num_cols:
    :return:
    """
    for f1 in tqdm(cat_cols):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_cols):
            tmp = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std'
            })
            df = df.merge(tmp, on=f1, how='left')
            del tmp
            gc.collect()
    return df


def cat_num_stats(df, cat_cols, num_cols):
    """
    类别特征与数据特征groupby统计特征, 复杂版
    :param df:
    :param cat_cols:
    :param num_cols:
    :return:
    """
    def max_min(x):
        return x.max() - x.min()

    def q10(x):
        return x.quantile(0.1)

    def q20(x):
        return x.quantile(0.2)

    def q30(x):
        return x.quantile(0.3)

    def q40(x):
        return x.quantile(0.4)

    def q60(x):
        return x.quantile(0.6)

    def q70(x):
        return x.quantile(0.7)

    def q80(x):
        return x.quantile(0.8)

    def q90(x):
        return x.quantile(0.9)

    for f1 in tqdm(cat_cols):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_cols):
            tmp = g[f2].agg({
                '{}_{}_cnt'.format(f1, f2): 'count',
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_mode'.format(f1, f2): lambda x: np.mean(pd.Series.mode(x)),
                # '{}_{}_mode'.format(f1, f2): lambda x: stats.mode(x)[0][0],
                # '{}_{}_mode'.format(f1, f2): lambda x: x.value_counts().index[0],
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_skew'.format(f1, f2): 'skew',
                '{}_{}_std'.format(f1, f2): 'std',
                '{}_{}_nunique'.format(f1, f2): 'nunique',
                '{}_{}_max_min'.format(f1, f2): lambda x: max_min(x),
                '{}_{}_q_10'.format(f1, f2): lambda x: q10(x),
                '{}_{}_q_20'.format(f1, f2): lambda x: q20(x),
                '{}_{}_q_30'.format(f1, f2): lambda x: q30(x),
                '{}_{}_q_40'.format(f1, f2): lambda x: q40(x),
                '{}_{}_q_60'.format(f1, f2): lambda x: q60(x),
                '{}_{}_q_70'.format(f1, f2): lambda x: q70(x),
                '{}_{}_q_80'.format(f1, f2): lambda x: q80(x),
                '{}_{}_q_90'.format(f1, f2): lambda x: q90(x),

            })
            df = df.merge(tmp, on=f1, how='left')
            del tmp
            gc.collect()
    return df


def topN(df, group_col, cal_col, N):
    """
    最受欢迎的元素及其频次
    :param df:
    :param group_col:
    :param cal_col:
    :param N: 欢迎程度, 0, 1, 2
    :return:
    """
    tmp = df.groupby(group_col, as_index=False)[cal_col].agg({
        '{}_{}_top_{}'.format(group_col, cal_col, N): lambda x: x.value_counts().index[N],
        '{}_{}_top_{}_cnt'.format(group_col, cal_col, N): lambda x: x.value_counts().values[N],
    })
    df = df.merge(tmp, on=group_col, how='left')
    del tmp
    gc.collect()
    return df


def pivot(df, index, columns, func):
    df['tmp'] = 1
    tmp = df.pivot_table(values='tmp', index=index, columns=columns, aggfunc=func).fillna(0)
    tmp.columns = ['{}_{}'.format(columns, f) for f in tmp.columns]
    tmp.reset_index(inplace=True)
    return tmp


# 获取TOP频次的位置信息，这里选Top3
mode_df = df.groupby(['ID', 'lat', 'lon'], as_index=False)['time'].agg({'mode_cnt': 'count'})
mode_df['rank'] = mode_df.groupby('ID')['mode_cnt'].rank(method='first', ascending=False)
for i in range(1, 4):
    tmp_df = mode_df[mode_df['rank'] == i]
    del tmp_df['rank']
    tmp_df.columns = ['ID', 'rank{}_mode_lat'.format(i), 'rank{}_mode_lon'.format(i), 'rank{}_mode_cnt'.format(i)]
    group_df = group_df.merge(tmp_df, on='ID', how='left')


def fe_stat(df):
    for c1 in tqdm(['author', 'level1', 'level2', 'level3', 'level4', 'brand', 'mall','author_brand','author_brand_mall', 'author_l1', 'author_l1-2','author_l1-3', 'author_l1-4', 'author_mall','url', 'baike_id_1h','baike_id_2h','date','week']):
        tr0 = df.groupby(c1)[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].agg(['count','nunique','sum','mean','median','max','min','std'])
        tr0.columns = [f'{x}_gp_{c1}_{y}_0810' for x,y in tr0.columns]
        df = pd.merge(df,tr0,left_on=c1,right_index=True,how='left')
        if c1 not in ['date','week']:
            tr0 = df.groupby([c1,'date'])[['orders_2h','orders_1h','price','price_diff','orders_21_diff', 'orders_21_rate']].agg(['count','nunique','sum','mean','median','max','min','std'])
            tr0.columns = [f'{x}_gp_{c1}date_{y}_0810' for x,y in tr0.columns]
            df = pd.merge(df,tr0,left_on=[c1,'date'],right_index=True,how='left')
    return df
