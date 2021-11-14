# Author: LC
# Date Time: 2021/11/13 19:34
# File Description: 一些乱七八糟的绘图

# 阈值与连通子图个数对应关系示意图
import matplotlib.pyplot as plt

import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.arange(-2, 2, 0.1)
y = np.exp(3 * -x)

plt.figure(figsize=(3.5, 2.5))
plt.plot(x, y)
plt.xlabel(u"阈值")
plt.ylabel(u"连通子图的个数")
plt.scatter([-1.2], [np.exp(3.6)])
plt.annotate('A', xy=(-1.2, np.exp(-1.2 * 3)), xytext=(-1.0, np.exp(-1.2 * 3) + 50))
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.grid(True)
plt.show()

