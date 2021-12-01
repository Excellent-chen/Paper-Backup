# Author: LC
# Date Time: 2021/11/13 19:34
# File Description: 一些乱七八糟的绘图

# # 阈值与连通子图个数对应关系示意图
# import matplotlib.pyplot as plt
#
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False
#
# x = np.arange(-2, 2, 0.1)
# y = np.exp(3 * -x)
#
# plt.figure(figsize=(3.5, 2.5))
# plt.plot(x, y, color='black', linewidth=0.5)
# plt.xlabel("阈值", fontsize=14)
# plt.ylabel("连通子图的个数", fontsize=14)
# plt.scatter([-1.0], [np.exp(3.0)], color='black', s=5)
# plt.annotate('A', xy=(-1.0, np.exp(-1.0 * 3)), xytext=(-1.0, np.exp(-1.0 * 3) + 60))
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.grid(True)
# plt.show()

# # 阈值与连通子图个数对应关系示意图
# import matplotlib.pyplot as plt
#
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False
#
# x = np.arange(-2, 2, 0.1)
# y = -np.exp(3 * -x)
#
# plt.figure(figsize=(3.5, 2.5))
# plt.plot(x, y, color='black', linewidth=0.5)
# plt.xlabel(u"阈值")
# plt.ylabel(u"最大连通子图节点个数")
# plt.scatter([-1.0], [-np.exp(3.0)], color='black', s=5)
# plt.annotate('B', xy=(-1.0, np.exp(-1.0 * 3)), xytext=(-1.0, np.exp(-1.0 * 3) - 60))
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.grid(True)
# plt.show()


# # DTW 对齐
# import matplotlib.pyplot as plt
#
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(figsize=(8, 2.5))
#
# y1 = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 2, 1, 1, 0, 0, 0, 0]).reshape((-1, 1))
#
# y2 = np.array([3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 5, 5, 4, 4, 3, 3]).reshape((-1, 1))
#
# path = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 10),
#         (9, 11), (10, 12), (11, 13), (12, 14), (13, 14), (14, 14), (15, 15)]
#
# plt.plot(y1, "black", label='序列 1', linewidth=1)
# plt.plot(y2, "black", label='序列 2', linewidth=1)
#
# for positions in path:
#     plt.plot([positions[0], positions[1]],
#              [y1[positions[0]], y2[positions[1]]], color='black', linestyle='dotted', linewidth=0.8)
#
# plt.xticks([])
# plt.yticks([])
#
# plt.axis('off')
#
# plt.legend()
# plt.show()


# # 贵州茅台；五粮液
# # 天级数据
# import tushare as ts
#
# from matplotlib import pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False
#
# ts.set_token('5e440fc23c7094ffebec94e06607adaf3a47cb337c6aeb63ba5fad71')
#
# df1 = ts.pro_api().daily(ts_code='000858.SZ,600519.SH', start_date='20201028', end_date='20211025')
#
# df1.sort_values(by=['trade_date'], inplace=True)
#
# y1 = df1.loc[df1['ts_code'] == '000858.SZ', ['pct_chg']].values.tolist()
# y2 = df1.loc[df1['ts_code'] == '600519.SH', ['pct_chg']].values.tolist()
#
# x = [i for i in range(len(y1))]
#
# plt.figure(figsize=(10, 2.5))
# plt.plot(x, y1, label='五粮液', linewidth=1, linestyle='solid')
# plt.plot(x, y2, label='贵州茅台', linewidth=1, linestyle='dashed')
# plt.xlabel(u'相对交易日（天）', fontsize=14)
# plt.ylabel(u'收益率（%）', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
#
# # 分钟级数据
# from utils.common_function import *
#
# db = get_db()
#
# cursor = db.cursor()
#
# y3 = []
#
# try:
#     sql = '''SELECT price FROM `minute_level_data_sh` WHERE `code` = '600519' AND date >= '2021-10-22 15:00:00' AND date < '2021-10-26' ORDER BY date'''
#     cursor.execute(sql)
#     prices = []
#     for price in cursor.fetchall():
#         prices.append(price[0])
#     for i in range(1, len(prices)):
#         y3.append((prices[i] - prices[i - 1]) / prices[i - 1] * 100)
# except Exception as e:
#     print(e)
#
# y4 = []
#
# try:
#     sql = '''SELECT price FROM `minute_level_data_sz` WHERE `code` = '000858' AND date >= '2021-10-22 15:00:00' AND date < '2021-10-26' ORDER BY date'''
#     cursor.execute(sql)
#     prices = []
#     for price in cursor.fetchall():
#         prices.append(price[0])
#     for i in range(1, len(prices)):
#         y4.append((prices[i] - prices[i - 1]) / prices[i - 1] * 100)
# except Exception as e:
#     print(e)
#
# cursor.close()
#
# db.close()
#
# plt.figure(figsize=(10, 2.5))
# plt.plot(x, y4, label='五粮液', linewidth=1, linestyle='solid')
# plt.plot(x, y3, label='贵州茅台', linewidth=1, linestyle='dashed')
# plt.xlabel('相对交易分钟（分）', fontsize=14)
# plt.ylabel('收益率（%）', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()

# import tushare as ts
#
# from matplotlib import pyplot as plt
# from matplotlib import ticker
#
# pro = ts.pro_api('5e440fc23c7094ffebec94e06607adaf3a47cb337c6aeb63ba5fad71')
#
# df1 = pro.index_daily(ts_code='000001.SH', start_date='20210311', end_date='20210901')  # 上证指数
#
# df1.sort_values(by=['trade_date'], inplace=True)
#
# x = df1['trade_date'].values.tolist()
#
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(figsize=(5, 3.5))
# plt.plot(x, df1['close'].values.tolist(), linewidth='1')
# ax = plt.gca()
# xml = ticker.MultipleLocator(len(x) // 8)
# ax.xaxis.set_major_locator(xml)
# plt.xticks(rotation=30)
# plt.xlabel('日期', fontsize=14)
# plt.ylabel('收益指数值', fontsize=14)
# plt.show()

import tushare as ts

from matplotlib import pyplot as plt
from matplotlib import ticker

pro = ts.pro_api('5e440fc23c7094ffebec94e06607adaf3a47cb337c6aeb63ba5fad71')

df1 = pro.index_daily(ts_code='000001.SH', start_date='20190901', end_date='20210901')  # 上证指数

df2 = pro.index_daily(ts_code='399001.SZ', start_date='20190901', end_date='20210901')  # 深证成指

df3 = pro.index_daily(ts_code='399300.SZ', start_date='20190901', end_date='20210901')  # 沪深300

df1.sort_values(by=['trade_date'], inplace=True)

df2.sort_values(by=['trade_date'], inplace=True)

df3.sort_values(by=['trade_date'], inplace=True)

x = df1['trade_date'].values.tolist()

plt.rcParams['font.sans-serif'] = ['simsun']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(5, 3.5))
plt.plot(x, df1['close'].values.tolist(), label='上证指数', linewidth='1', linestyle='solid')
plt.plot(x, df3['close'].values.tolist(), label='沪深300', linewidth='1', linestyle='dashed')
ax = plt.gca()
xml = ticker.MultipleLocator(len(x) // 8)
ax.xaxis.set_major_locator(xml)
plt.xticks(rotation=30)
plt.xlabel('日期', fontsize=14)
plt.ylabel('收益指数值', fontsize=14)
plt.legend(fontsize=14)
plt.show()

