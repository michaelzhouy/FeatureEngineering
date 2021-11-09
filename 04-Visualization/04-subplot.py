# -*- coding: utf-8 -*-
# @Time    : 2021/9/4 12:19 下午
# @Author  : Michael Zhouy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.DataFrame()
# grouped_df = df.groupby(['day', 'hour'])['is_trade'].aggregate('mean').reset_index()
# grouped_df = grouped_df.pivot('day', 'hour', 'is_trade')
# plt.figure(figsize=(12, 6))
# sns.heatmap(grouped_df)
# plt.title("CVR of Day Vs Hour")
# plt.show()


x = [1, 2, 3, 4, 5]
y = [3, 6, 7, 9, 2]
# fig, ax=plt.subplots(1,2)
plt.figure(1)
plt.subplot(121)  # 12表示子图分布:一行2列；最后一个1表示第1个子图，从左往右
plt.plot(x, y, label='trend')
plt.title('title 1', fontsize=12, color='r')  # r: red
plt.subplot(122)  # 第二个子图
plt.plot(x, y, c='cyan')
plt.title('title 2')
plt.show()
