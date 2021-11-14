# Author: LC
# Date Time: 2021/11/5 14:44
# File Description:

from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw

from utils.evaluation import *
from utils.metric import *


# Step 1: 获取成分股基本信息
def fetch_cons_base_info():
    # 数据来源 - 中证指数有限公司:
    #   https://www.csindex.com.cn/#/indices/family/detail?indexCode=000300
    cons_base_info = pd.read_excel('./data/000300cons.xls', dtype=str)
    return cons_base_info


# Step 2: 获取成分股价格信息
def fetch_cons_price_info(cons_base_info):
    # from tqdm import tqdm
    # cons_price_info, trade_date = pd.DataFrame(), fetch_trade_date()[-120:]
    # for cons_code, exchange in tqdm(cons_base_info[['成分券代码Constituent Code', '交易所英文名称Exchange(Eng)']].values.tolist()):
    #     ts_code = cons_code + '.' + ('SZ' if 'Shenzhen' in exchange else 'SH')
    #     price_info = pro.daily(ts_code=ts_code, start_date=trade_date[0], end_date=trade_date[-1])[
    #         ['ts_code', 'trade_date', 'pct_chg']]
    #     cons_price_info = pd.concat([cons_price_info, price_info])
    # cons_price_info.sort_values(by=['trade_date', 'ts_code'], inplace=True)
    # cons_price_info.to_pickle('./data/cons_price_info.pickle')
    cons_price_info = pd.read_pickle('./data/cons_price_info.pickle')
    return cons_price_info


# 股票距离度量函数
def fetch_cons_distance(cons_price_info, N):
    # cons_price_info = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    # # 计算不同股票涨跌幅序列之间的相关性
    # cons_distance = []
    # from itertools import combinations_with_replacement
    # for code1, code2 in combinations_with_replacement(cons_price_info.keys(), 2):
    #     info1, info2 = cons_price_info[code1], cons_price_info[code2]
    #     cons_distance.append([code1, code2, distance_measure(info1, info2, return_path=False)[0]])
    #     if code1 != code2:
    #         cons_distance.append([code2, code1, distance_measure(info2, info1, return_path=False)[0]])
    # #
    # cons_distance = pd.DataFrame(data=cons_distance, columns=['code1', 'code2', 'distance'])
    # cons_distance.to_pickle('./data/cons_distance_' + str(N) + '.pickle')
    # return cons_distance
    if N == 30:
        return pd.read_pickle('./data/cons_distance_30.pickle')
    elif N == 60:
        return pd.read_pickle('./data/cons_distance_60.pickle')
    elif N == 120:
        return pd.read_pickle('./data/cons_distance_120.pickle')


# Step 5: 获取合适的聚类参数，包括类别数量和初始中心
def fetch_cluster_parm(cons_price_info, cons_distance):
    # # Step 3.1: 获取阈值与连通子图个数之间的对应关系
    # cons_price_info = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    # threshold_info, lower, upper = fetch_threshold_info(cons_price_info, cons_distance)
    # from matplotlib import pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['simsun']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(5, 3.5))
    # plt.plot(threshold_info['threshold'], threshold_info['count'], linewidth=2)
    # plt.scatter([1.11], [9.1], s=8)
    # plt.annotate('A', xy=(1.11, 9.1), xytext=(1.21, 11))
    # plt.xticks(np.arange(lower, upper + 0.01, 2))
    # plt.yticks(np.arange(0, 300 + 5, 50))
    # plt.xlabel('阈值')
    # plt.ylabel('连通子图个数')
    # plt.xlim((0, upper))
    # plt.ylim((0, 300))
    # plt.show()
    # return None
    # return threshold_info['centroids'][23]
    return ['300274.SZ', '002460.SZ', '600111.SH', '603799.SH', '600362.SH', '603501.SH', '601012.SH', '000625.SZ',
            '601877.SH']
    # K值选取对比实验
    # return ['300274.SZ', '600111.SH', '601877.SH']
    # return ['300274.SZ', '002460.SZ', '600111.SH', '603799.SH', '600362.SH', '603501.SH', '601012.SH', '000625.SZ', '601877.SH']
    # return ['601877.SH', '601390.SH', '600111.SH', '688169.SH', '300274.SZ', '002179.SZ', '000776.SZ', '600926.SH',
    #         '002601.SZ', '600703.SH', '601688.SH', '603501.SH', '603799.SH', '002460.SZ', '601012.SH']
    # return ['601877.SH', '600111.SH', '002129.SZ', '603806.SH', '688169.SH', '300274.SZ', '600893.SH', '002179.SZ',
    #         '000776.SZ', '600926.SH', '600438.SH', '002601.SZ', '600585.SH', '600703.SH', '000568.SZ', '603939.SH',
    #         '603501.SH', '601800.SH', '603799.SH', '002460.SZ', '601012.SH']
    # # 初始点选取对比实验
    # import random
    # return random.sample(list(set(cons_distance['code1'])), 9)


# Step 6: 改进版K-Medoids
def MKMedoids(cons_price_info, centroids, N, max_iter=50):
    # cons_price_info = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    # #
    # n_cluster = len(centroids)
    # #
    # codes = list(cons_price_info.keys())
    # codes.sort()
    # # 辅助数据结构，用于存储每只股票对应的簇编号以及与簇之间的路径
    # cons_cluster_info = [[codes[i], -1, []] for i in range(len(codes))]
    # # 获取初始质心对应的时间序列
    # centroid_series = [0] * n_cluster
    # for i in range(len(centroids)):
    #     centroid_series[i] = cons_price_info[centroids[i]]
    # #
    # clusterChanged, iter = True, 0
    # while clusterChanged and iter <= max_iter:
    #     clusterChanged = False
    #     for i in range(len(codes)):
    #         minDist, minIndex, minPath = float('inf'), -1, []
    #         for k in range(n_cluster):
    #             dist, path = distance_measure(cons_price_info[codes[i]], centroid_series[k])
    #             if dist < minDist:
    #                 minDist, minIndex, minPath = dist, k, path
    #         if cons_cluster_info[i][1] != minIndex:
    #             clusterChanged = True
    #             cons_cluster_info[i][1] = minIndex
    #             cons_cluster_info[i][2] = minPath
    #     # 重新计算质心
    #     new_cons_price_info = []
    #     for i in range(n_cluster):
    #         new_cons_price_info.append([[] for _ in range(len(centroid_series[i]))])
    #     for cons, index, path in cons_cluster_info:
    #         for x, y in path:
    #             new_cons_price_info[index][y].append(cons_price_info[cons][x])
    #     for i in range(n_cluster):
    #         for j in range(len(centroid_series[i])):
    #             # 簇更新策略对比实验
    #             k = len(new_cons_price_info[i][j]) // 2
    #             new_cons_price_info[i][j].sort()
    #             if 2 * k == len(new_cons_price_info[i][j]):
    #                 centroid_series[i][j] = (new_cons_price_info[i][j][k - 1] + new_cons_price_info[i][j][k]) / 2
    #             else:
    #                 centroid_series[i][j] = new_cons_price_info[i][j][k]
    #             # centroid_series[i][j] = np.mean(new_cons_price_info[i][j])  # 平均值
    #             # import random
    #             # centroid_series[i][j] = random.sample(new_cons_price_info[i][j], 1)[0]  # 随机
    #     iter += 1
    # cons_cluster_info = pd.DataFrame(data=cons_cluster_info, columns=['cons', 'cluster', 'path'])
    # cons_cluster_info.to_pickle('./data/cons_cluster_info_' + str(N) + '.pickle')
    cons_cluster_info = pd.read_pickle('./data/cons_cluster_info_' + str(N) + '.pickle')
    return cons_cluster_info


# Step 7: 对聚类结果进行可视化
def visualization(cons_price_info, cons_cluster_info, N):
    # cons_price_info = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    # from matplotlib import pyplot as plt
    # x = [i for i in range(N)]
    # info = cons_cluster_info.loc[cons_cluster_info['cluster'] == 0]
    # plt.figure(figsize=(8, 2))
    # for cons in info['cons'].values.tolist():
    #     if len(cons_price_info[cons]) != N:
    #         continue
    #     plt.plot(x, cons_price_info[cons], linewidth=1)
    # plt.show()
    # info = cons_cluster_info.loc[cons_cluster_info['cluster'] == 3]
    # plt.figure(figsize=(8, 2))
    # for cons in info['cons'].values.tolist():
    #     if len(cons_price_info[cons]) != N:
    #         continue
    #     plt.plot(x, cons_price_info[cons], linewidth=1)
    # plt.show()
    from matplotlib import pyplot as plt
    x = [i for i in range(N)]
    data = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    for cluster, info in cons_cluster_info.groupby(by=['cluster']):
        plt.figure(figsize=(8, 2))
        for cons in info['cons'].values.tolist():
            if len(data[cons]) != N:
                continue
            plt.plot(x, data[cons])
        plt.title(str(N) + str(cluster))
        plt.show()


# Step 8: 对聚类结果进行评估
def evaluation(cons_price_info, cons_cluster_info, cons_distance):
    cons_cluster_info = cons_cluster_info.rename(columns={'id': 'cluster'})
    # Proposed
    data = pd.merge(cons_price_info, cons_cluster_info, left_on=['ts_code'], right_on=['cons'])
    data = data.rename(columns={'cons': 'code', 'cluster': 'id'})
    asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
    sdrdc = fetch_sdrdc(asdrc, ardc)
    ss = fetch_ss(cons_distance, cons_cluster_info)
    print('Proposed ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f. SS:%.4f.' % (asdrc, ardc, sdrdc, ss))


def comparison(cons_price_info, N, n_clusters):
    price_info = {code: info['pct_chg'].values.tolist() for code, info in cons_price_info.groupby(by=['ts_code'])}
    codes, dataset = [], []
    for code, info in price_info.items():
        if len(info) != N:
            continue
        codes.append(code)
        dataset.append(info)
    dataset = np.array(dataset)
    # E K-Means
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean', max_iter=50, random_state=0)
    label = model.fit_predict(dataset)
    cons_cluster_info = []
    for code, id in zip(codes, label):
        cons_cluster_info.append([code, id])
    cons_cluster_info = pd.DataFrame(data=cons_cluster_info, columns=['code', 'id'])
    data = pd.merge(cons_price_info, cons_cluster_info, left_on=['ts_code'], right_on=['code'])
    asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
    sdrdc = fetch_sdrdc(asdrc, ardc)
    ss = silhouette_score(dataset, label, metric='euclidean')
    print('E K-Means ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f. SS:%.4f.' % (asdrc, ardc, sdrdc, ss))
    # D K-Means
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', max_iter=50, random_state=0)
    label = model.fit_predict(dataset)
    cons_cluster_info = []
    for code, id in zip(codes, label):
        cons_cluster_info.append([code, id])
    cons_cluster_info = pd.DataFrame(data=cons_cluster_info, columns=['code', 'id'])
    data = pd.merge(cons_price_info, cons_cluster_info, left_on=['ts_code'], right_on=['code'])
    asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
    sdrdc = fetch_sdrdc(asdrc, ardc)
    ss = silhouette_score(dataset, label, metric='dtw')
    print('D K-Means ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f. SS:%.4f.' % (asdrc, ardc, sdrdc, ss))
    # E K-Medoids
    model = KMedoids(n_clusters=n_clusters, metric='euclidean', max_iter=50, random_state=0)
    label = model.fit_predict(dataset)
    cons_cluster_info = []
    for code, id in zip(codes, label):
        cons_cluster_info.append([code, id])
    cons_cluster_info = pd.DataFrame(data=cons_cluster_info, columns=['code', 'id'])
    data = pd.merge(cons_price_info, cons_cluster_info, left_on=['ts_code'], right_on=['code'])
    asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
    sdrdc = fetch_sdrdc(asdrc, ardc)
    ss = silhouette_score(dataset, label, metric='euclidean')
    print('E K-Medoids ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f. SS:%.4f.' % (asdrc, ardc, sdrdc, ss))
    # D K-Medoids
    model = KMedoids(n_clusters=n_clusters, metric='precomputed', max_iter=50, random_state=0)
    dist = cdist_dtw(dataset)
    label = model.fit_predict(dist)
    cons_cluster_info = []
    for code, id in zip(codes, label):
        cons_cluster_info.append([code, id])
    cons_cluster_info = pd.DataFrame(data=cons_cluster_info, columns=['code', 'id'])
    data = pd.merge(cons_price_info, cons_cluster_info, left_on=['ts_code'], right_on=['code'])
    asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
    sdrdc = fetch_sdrdc(asdrc, ardc)
    ss = silhouette_score(dist, label, metric='precomputed')
    print('D K-Medoids ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f. SS:%.4f.' % (asdrc, ardc, sdrdc, ss))


if __name__ == '__main__':
    # Step 1: 获取成分股基本信息
    cons_base_info = fetch_cons_base_info()
    # Step 2: 获取成分股价格信息
    cons_price_info = fetch_cons_price_info(cons_base_info)
    # Step 3: 分时间区间进行实验
    # for start_date, N in (('20210311', 120),):
    for start_date, N in (('20210722', 30), ('20210609', 60), ('20210311', 120)):
        #
        print('N:%d.' % N)
        #
        price_info = cons_price_info.loc[cons_price_info['trade_date'] >= start_date]
        # Step 4: 获取成分股间的距离
        cons_distance = fetch_cons_distance(price_info, N)
        # Step 5: 获取合适的聚类参数
        cluster_parm = fetch_cluster_parm(price_info, cons_distance)
        # Step 6: 利用改进版K-Medoids对成分股进行聚类
        # cons_cluster_info = None
        cons_cluster_info = MKMedoids(price_info, cluster_parm, N)
        # Step 7: 对聚类结果进行可视化
        # visualization(price_info, cons_cluster_info, N)
        # Step 8: 对聚类结果进行评估
        evaluation(price_info, cons_cluster_info, cons_distance)
        # Step 9: 其它聚类结果对比
        comparison(price_info, N, len(cluster_parm))
