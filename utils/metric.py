# Author: LC
# Date Time: 2021/11/5 15:51
# File Description:

import numpy as np


# 距离函数
def distance(x, y):
    return abs(x - y)


def distance_measure(X, Y, beta=0.6, punish=0.8, return_path=True, convert=False):
    """
    :param X: 时间序列 X
    :param Y: 时间序列 Y
    :param beta: 权重系数
    :param punish: 乘法系数
    :return:
    """
    m, n = len(X), len(Y)
    d = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            d[i][j] = distance(X[i], Y[j])
    D = [[0] * n for _ in range(m)]
    D[0][0] = d[0][0]
    for i in range(1, m):
        D[i][0] = D[i - 1][0] + d[i][0]
    for i in range(1, n):
        D[0][i] = D[0][i - 1] + d[0][i]
    for i in range(1, m):
        for j in range(1, n):
            D[i][j] = beta * d[i][j] + (1 - beta) * min(D[i - 1][j - 1], D[i][j - 1] / punish, D[i - 1][j])
    path = []
    if return_path:
        path = [(m - 1, n - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j - 1))
            elif j == 0:
                path.append((i - 1, 0))
            else:
                index = np.argmin(np.array([D[i - 1][j - 1], D[i][j - 1], D[i - 1][j]]))
                if index == 0:
                    path.append((i - 1, j - 1))
                elif index == 1:
                    path.append((i, j - 1))
                else:
                    path.append((i - 1, j))
    return (1 / (1 + D[-1][-1])) if convert else D[-1][-1], path[::-1]


class UnionFind:
    # 初始化函数
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n

    # 根节点查询函数
    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    # 根节点合并函数
    def merge(self, x, y):
        rootX, rootY = self.find(x), self.find(y)
        if rootX == rootY:
            return
        else:
            self.root[rootY] = rootX
            self.size[rootX] += self.size[rootY]

    # 获取并查集基本信息
    def fetch_union_info(self, num2code):
        # import pandas as pd
        # k, data, code2label = 0, [], {}
        centroids, max_size, count = [], 0, 0
        for i in range(len(self.root)):
            if self.root[i] == i and self.size[i] > max_size:
                max_size = self.size[i]
            # rootI = self.find(i)
            # if num2code[rootI] not in code2label:
            #     code2label[num2code[rootI]] = k
            #     k += 1
            # data.append([num2code[i], code2label[num2code[rootI]]])
            if self.root[i] == i:
                centroids.append(num2code[i])
                count += 1
        # if count in (9, 15, 21):
        #     data = pd.DataFrame(data=data, columns=['cons', 'cluster'])
        #     data.to_pickle('./data/cons_cluster_info_' + str(count) + '_2.pickle')
        return centroids, max_size, count


def fetch_union_info(cons_price_info, data):
    codes = set(cons_price_info.keys())
    code2num = {code: i for i, code in enumerate(codes)}
    num2code = {i: code for code, i in code2num.items()}
    #
    n = len(codes)
    unionfind = UnionFind(n)
    for code1, code2, _ in data.values.tolist():
        unionfind.merge(code2num[code1], code2num[code2])
    return unionfind.fetch_union_info(num2code)


def fetch_threshold_info(cons_price_info, cons_distance, h=0.05):
    threshold_info = {'threshold': [], 'centroids': [], 'maxsize': [], 'count': []}
    lower, upper = min(cons_distance['distance']), max(cons_distance['distance'])
    for threshold in np.arange(lower, upper, h):
        # 筛选数据
        data = cons_distance.loc[cons_distance['distance'] <= threshold]
        # 最大连通子图节点个数
        centroids, max_size, count = fetch_union_info(cons_price_info, data)
        threshold_info['threshold'].append(threshold)
        threshold_info['centroids'].append(centroids)
        threshold_info['maxsize'].append(max_size)
        threshold_info['count'].append(count)
    return threshold_info, lower, upper
