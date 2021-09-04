# -*- coding: utf-8 -*-
# @Time    : 2021/9/4 12:19 下午
# @Author  : Michael Zhouy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame()

grouped_df = df.groupby(['day', 'hour'])['is_trade'].aggregate('mean').reset_index()
grouped_df = grouped_df.pivot('day', 'hour', 'is_trade')
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_df)
plt.title("CVR of Day Vs Hour")
plt.show()
