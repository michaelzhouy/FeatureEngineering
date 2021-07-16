# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 8:18 下午
# @Author  : Michael Zhouy
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import pyhdfs
from impala import dbapi


def get_partitions(hdfs_dir):
    client = pyhdfs.HdfsClient(hosts='10.21.3.10,9000', user_name='hadoop')
    dirs, partitions = [], []
    if client.exists(hdfs_dir):
        for file1 in client.listdir(hdfs_dir):
            hdfs_dir2 = hdfs_dir + file1
            # print('222: ', hdfs_dir2)
            for file2 in client.listdir(hdfs_dir2):
                hdfs_dir3 = hdfs_dir + file1 + '/' + file2
                # print('333: ', hdfs_dir3)
                for file3 in client.listdir(hdfs_dir3):
                    hdfs_dir4 = hdfs_dir + file1 + '/' + file2 + '/' + file3
                    dirs.append(hdfs_dir4)
                    # print('444: ', hdfs_dir4)
    for dir in dirs:
        dt = dir.split('/')[-3].split('=')[-1]
        area = dir.split('/')[-2].split('=')[-1]
        y = dir.split('/')[-1].split('=')[-1]
        partitions.append("dt_part='{}', area_part='{}', y_part='{}'".format(dt, area, y))
    return partitions


def alter_partitions(partitions, table):
    conn = dbapi.connect(
        host='10.21.3.22',
        port=10000,
        auth_mechanism='PLAIN',
        database='writing_height'
    )
    cursor = conn.cursor()

    # alter table x_direction
    # add partition (dt_part='2021-07-15', area_part='1', y_part='2139')
    for partition in partitions:
        sql = """alter table {} add partition ({})""".format(table, partition)
        try:
            cursor.execute(sql)
        except:
            print(sql)


table = 'x_direction'
hdfs_dir = '/user/hive/warehouse/writing_height.db/{}/'.format(table)

partitions = get_partitions(hdfs_dir)
alter_partitions(partitions, table)
