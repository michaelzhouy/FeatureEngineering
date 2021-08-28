# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 11:07 上午
# @Author  : Michael Zhouy
import pandas as pd


def to_excel(df1, df2, path):
    writer = pd.ExcelWriter(path)
    df1.to_excel(writer, sheet_name='31', startrow=0, index=None)
    df2.to_excel(writer, sheet_name='31', startrow=0, index=None)

    writer.save()
    writer.close()
