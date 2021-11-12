# Author: LC
# Date Time: 2021/11/5 14:09
# File Description:

import numpy as np
import pandas as pd

from tslearn.clustering import silhouette_score


# 平均类内同向收益比
def fetch_asdrc(data):
    # Step 01: 计算每一天每个类别对应的类内同向收益率
    rec = []
    for (date, id), subdata in data.groupby(by=['trade_date', 'id']):
        n = subdata.shape[0]
        cnt_pos = subdata.loc[subdata['pct_chg'] > 0].shape[0]
        cnt_neg = subdata.loc[subdata['pct_chg'] < 0].shape[0]
        SDRC = max(cnt_pos, cnt_neg, n - cnt_pos - cnt_neg) / n
        rec.append([date, id, SDRC])
    rec = pd.DataFrame(data=rec, columns=['date', 'id', 'sdrc'])
    # Step 02: 计算每一天所有类别对应的平均类内同向收益率
    asdrc, t = 0, 0
    for date, subdata in rec.groupby(by=['date']):
        t += 1
        m = subdata.shape[0]
        asdrc += sum(subdata['sdrc']) / m
    return asdrc / t


# 平均类内收益率偏离度
def fetch_ardc(data):
    # Step 01: 计算每一天每个类别对应的类内收益率偏离度
    rec = []
    for (date, id), subdata in data.groupby(by=['trade_date', 'id']):
        rdc = np.var(subdata['pct_chg'].values)
        rec.append([date, id, rdc])
    rec = pd.DataFrame(data=rec, columns=['date', 'id', 'rdc'])
    # Step 02: 计算每一天所有类别对应的平均类内收益率偏离度
    ardc, t = 0, 0
    for date, subdata in rec.groupby(by=['date']):
        t += 1
        m = subdata.shape[0]
        ardc += sum(subdata['rdc']) / m
    return ardc / t


# ASDRC、ARDC融合纠正指标
def fetch_sdrdc(asdrc, ardc, alpha=0.01, beta=0, m=9):
    return alpha * ardc * (beta + m) * (beta + m) / asdrc


# 轮廓系数
def fetch_ss(cons_distance, cons_cluster_info):
    codes, labels = cons_cluster_info['cons'].values.tolist(), cons_cluster_info['cluster'].values.tolist()
    code2num = {code: i for i, code in enumerate(codes)}
    dist = [[0] * len(codes) for _ in range(len(codes))]
    for code1, code2, distance in cons_distance.values.tolist():
        dist[code2num[code1]][code2num[code2]] = distance
    return silhouette_score(dist, labels, metric='precomputed')
