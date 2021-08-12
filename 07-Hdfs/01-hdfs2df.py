# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 8:12 下午
# @Author  : Michael Zhouy
import pandas as pd
from impala import dbapi
from impala.util import as_pandas


def hdfs2df(sql, host, port, database):
    conn = dbapi.connect(
        host=host,
        port=port,
        auth_mechanism='PLAIN',
        database=database
    )
    cursor = conn.cursor()
    cursor.execute(sql)
    data = as_pandas(cursor)
    return data


def hdfs(sql, host, port, database):
    conn = dbapi.connect(
        host=host,
        port=port,
        auth_mechanism='PLAIN',
        database=database
    )
    df = pd.read_sql(sql, conn)
    return df
