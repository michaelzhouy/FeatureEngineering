# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 5:53 下午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
import os
import gc
import time
import shutil


# 读取压缩文件
df = pd.read_csv('../input/file.txt.gz', compression='gzip', header=0, sep=',', quotechar='"')

# 分块读取数据
reader = pd.read_csv('../input/train.csv', iterator=True)
df = reader.get_chunk(10000)

# 数据存储为h5格式
df.to_hdf('data.h5', 'df')
# 读取h5文件
pd.read_hdf('data.h5')

df.to_feather('data.feather')
df = pd.read_feather('data.feather')

df[['sid', 'label']].to_csv('../sub/sub_{}_1.csv'.format(time.strftime('%Y%m%d')), index=False)


# 节省内存读文件
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
            df[col] = df[col].astype('str')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def read_data(path):
    """
    读取文件下所有的文件，并合并成一个文件
    :param path:
    :return:
    """
    data_list = []
    for f in os.listdir(path):
        print(f)
        df = pd.read_csv(path + os.sep + f)
        print(df.shape)
        data_list.append(df)
        del df
        gc.collect()

    res = pd.concat(data_list, ignore_index=True)
    return res


def time_split(df):
    """
    根据时间划分训练集, 验证集, 测试集
    :param df:
    :return:
    """
    train = df.loc[df['observe_date'] < '2019-11-04', :]
    valid = df.loc[(df['observe_date'] >= '2019-11-04') & (df['observe_date'] <= '2019-12-04'), :]
    test = df.loc[df['observe_date'] > '2020-01-04', :]
    return train, valid, test


def copy_file(files, path1, path2):
    """
    copy文件
    :param files:
    :param path1:
    :param path2:
    :return:
    """
    for file in files:
        src = path1 + file
        dst = path2 + file
        shutil.copyfile(src, dst)
