# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 8:12 下午
# @Author  : Michael Zhouy
from impala import dbapi

conn = dbapi.connect(
    host='10.21.3.22',
    port=10000,
    auth_mechanism='PLAIN',
    database='default'
)
cursor = conn.cursor()
cursor.execute('select * from isc.replace_group_quantity_pred')
