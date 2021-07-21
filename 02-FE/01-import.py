# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 3:33 下午
# @Author  : Michael Zhouy
# !pip freeze > requirements.txt
# 过滤警告
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

df = pd.DataFrame()

# DataFrame显示所有列
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


# 筛选object特征
df_object = df.select_dtypes(include=['object'])
df_numerical = df.select_dtypes(exclude=['object'])
