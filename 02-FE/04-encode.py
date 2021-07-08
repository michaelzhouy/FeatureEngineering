# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:45 下午
# @Author  : Michael Zhouy
import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def low_freq_encode(df, cat_cols, freq=2):
    for i in cat_cols:
        name_dict = dict(zip(*np.unique(df[i], return_counts=True)))
        df['{}_low_freq'.format(i)] = df[i].apply(lambda x: -999 if name_dict[x] < freq else name_dict[x])
    return df


def count_encode(df, cat_cols):
    for col in cat_cols:
        print(col)
        vc = df[col].value_counts(dropna=True)
        df[col + '_count'] = df[col].apply(lambda x: -999 if vc[x] < 10 else vc[x])
    return df


def label_encode(df, cat_cols, verbose=True):
    """
    label encode
    :param df:
    :param cat_cols:
    :param verbose:
    :return:
    """
    for col in cat_cols:
        df[col], _ = df[col].factorize(sort=True)
        if df[col].max() > 32000:
            df[col] = df[col].astype('int32')
        else:
            df[col] = df[col].astype('int16')
        if verbose:
            print(col)
    return df


def get_same_set(train_df, test_df):
    """
    test中出现，train中没有出现的取值编码
    :param train_df:
    :param test_df:
    :return:
    """
    train_diff_test = set(train_df) - set(test_df)
    same = set(train_df) - train_diff_test
    test_diff_train = set(test_df) - same
    dic_ = {}
    cnt = 0
    for val in same:
        dic_[val] = cnt
        cnt += 1
    for val in train_diff_test:
        dic_[val] = cnt
        cnt += 1
    for val in test_diff_train:
        dic_[val] = cnt
        cnt += 1
    return dic_


train = pd.DataFrame()
test = pd.DataFrame()
data = pd.concat([train, test], ignore_index=True)

for col in data.columns:
    if col != 'id' and col != 'click':
        if data[col].dtypes == 'O':
            print(col)
            dic_ = get_same_set(train[col].values, test[col].values)
            data[col+'_zj_encode'] = data[col].map(lambda x: dic_[x])


def train_test_label_encode(df, cat_col, type='save', path='./'):
    """
    train和test分开label encode
    save的食用方法
    for i in cat_cols:
        train = train_test_label_encode(train, i, 'save', './')
        train[i] = train[i].astype('category')
    load的食用方法：
    for i in cat_cols:
        d = train_test_label_encode(test, i, 'load', '../train_code/')
        test[i] = test[i].map(d)
        test[i] = test[i].astype('category')
    :param df:
    :param cat_col:
    :param type:
    :param path:
    :return:
    """
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    if type == 'save':
        print(cat_col)
        d = dict(zip(df[cat_col].unique(), range(df[cat_col].nunique())))
        df[cat_col] = df[cat_col].map(d)
        np.save(path + '{}.npy'.format(cat_col), d)
        return df
    elif type == 'load':
        d = np.load(path + '{}.npy'.format(cat_col), allow_pickle=True).item()
        return d


def train_test_label_encode_2(df, cat_col, type='save', path='./'):
    """
    train和test分开label encode
    save的食用方法
    for i in cat_cols:
        train = train_test_label_encode_2(train, i, 'save', './')
        train[i] = train[i].astype('category')
    load的食用方法：
    for i in cat_cols:
        d = train_test_label_encode_2(test, i, 'load', '../train_code/')
        test[i] = test[i].map(d)
        test[i] = test[i].astype('category')
    :param df:
    :param cat_col:
    :param type: 'save' 'load'
    :param path:
    :return:
    """
    if type == 'save':
        print(cat_col)
        le = LabelEncoder()
        le.fit(df[cat_col])
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        df[cat_col] = df[cat_col].map(le_dict)
        np.save(path + '{}.npy'.format(cat_col), d)
        return df
    elif type == 'load':
        print(cat_col)
        d = np.load(path + '{}.npy'.format(cat_col), allow_pickle=True).item()
        return d
