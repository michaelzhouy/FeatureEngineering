# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:27 下午
# @Author  : Michael Zhouy
import pandas as pd


def binning(df, num_cols):
    """
    数值特征离散化
    @param df:
    @param num_cols:
    @return:
    """
    cat_cols = []
    for f in num_cols:
        for bins in [20, 50, 100, 200]:
            cat_cols.append('cut_{}_{}_bins'.format(f, bins))
            df['cut_{}_{}_bins'.format(f, bins)] = pd.cut(df[f], bins, duplicates='drop').apply(lambda x: x.left).astype(int)
    return df, cat_cols