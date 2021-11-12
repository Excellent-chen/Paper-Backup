# Author: LC
# Date Time: 2021/11/6 20:18
# File Description:

from collections import defaultdict
from itertools import combinations_with_replacement

from utils.common_function import *
from utils.metric import *


def fetch_stock_base_info():
    db = get_db()
    cursor = db.cursor()
    stock_base_info = pd.DataFrame()
    try:
        SQL = 'SELECT `code` FROM `semantic_attribute_contains` WHERE id = 61'
        cursor.execute(SQL)
        stock_base_info = pd.DataFrame(data=cursor.fetchall(), columns=['symbol'])
    except Exception as e:
        print('Fetch stock base info error: %s!' % e)
    finally:
        cursor.close()
        db.close()
        return stock_base_info


def fetch_stock_price_info(stock_base_info, start_date='20210301', end_date='20210901'):
    stock_price_info = pd.DataFrame()
    for ts_code in pd.merge(pro.stock_basic(), stock_base_info, on=['symbol'])['ts_code'].values:
        info = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        stock_price_info = pd.concat([stock_price_info, info])
    stock_price_info.sort_values(by=['trade_date', 'ts_code'], inplace=True)
    return stock_price_info


def fetch_stock_correlation_matrix(stock_price_info):
    stock_series = {code: info['pct_chg'].values.tolist() for code, info in stock_price_info.groupby(by=['ts_code'])}
    stock_correlation_matrix = {'E': [], 'D': [], 'T': []}
    for code1, code2 in combinations_with_replacement(stock_series.keys(), 2):
        series1, series2 = stock_series[code1], stock_series[code2]
        # 欧式距离
        if len(series1) == 128 and len(series2) == 128:  # 要求序列长度相等
            dist = np.sqrt(np.sum(np.square(np.array(series1) - np.array(series2)))) / len(series1)
            stock_correlation_matrix['E'].append([code1, code2, dist])
            if code1 != code2:
                stock_correlation_matrix['E'].append([code2, code1, dist])
        # 常规DTW距离
        dist, _ = distance_measure(series1, series2, beta=0.5, punish=1, return_path=False)
        stock_correlation_matrix['D'].append([code1, code2, 2 * dist])
        if code1 != code2:
            stock_correlation_matrix['D'].append([code2, code1, 2 * dist])
        # tw-lDTW距离
        dist, _ = distance_measure(series1, series2, beta=0.6, punish=0.8, return_path=False)
        stock_correlation_matrix['T'].append([code1, code2, dist])
        if code1 != code2:
            dist, _ = distance_measure(series2, series1, beta=0.6, punish=0.8, return_path=False)
            stock_correlation_matrix['T'].append([code2, code1, dist])
    columns = ['code1', 'code2', 'distance']
    stock_correlation_matrix['E'] = pd.DataFrame(data=stock_correlation_matrix['E'], columns=columns)
    stock_correlation_matrix['D'] = pd.DataFrame(data=stock_correlation_matrix['D'], columns=columns)
    stock_correlation_matrix['T'] = pd.DataFrame(data=stock_correlation_matrix['T'], columns=columns)
    return stock_correlation_matrix


def stock_visualization(stock_price_info, stock_correlation_matrix, base_code='000001.SZ', index=0):
    stock_series = {code: info['pct_chg'].values.tolist() for code, info in stock_price_info.groupby(by=['ts_code'])}
    x = [i for i in range(128)]
    y1 = stock_series[base_code]
    #
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    # E
    plt.figure(figsize=(10, 2.5))
    matrix_e = stock_correlation_matrix['E']
    info_e = matrix_e.loc[(matrix_e['code1'] == base_code) & (matrix_e['code2'] != base_code)]
    info_e.sort_values(by=['distance'], inplace=True)
    y2 = stock_series[info_e.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y2, label=info_e.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()
    # D
    plt.figure(figsize=(10, 2.5))
    matrix_d = stock_correlation_matrix['D']
    info_d = matrix_d.loc[(matrix_d['code1'] == base_code) & (matrix_d['code2'] != base_code)]
    info_d.sort_values(by=['distance'], inplace=True)
    y3 = stock_series[info_d.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y3, label=info_d.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()
    # T
    plt.figure(figsize=(10, 2.5))
    matrix_t = stock_correlation_matrix['T']
    info_t = matrix_t.loc[(matrix_t['code1'] == base_code) & (matrix_t['code2'] != base_code)]
    info_t.sort_values(by=['distance'], inplace=True)
    y4 = stock_series[info_t.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y4, label=info_t.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()


def fetch_threshold(stock_price_info, stock_correlation_matrix):
    stock_series = {code: info['pct_chg'].values.tolist() for code, info in stock_price_info.groupby(by=['ts_code'])}
    #
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    # E
    threshold_info_e, lower, upper = fetch_threshold_info(stock_series, stock_correlation_matrix['E'], h=0.01)
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_e['threshold'], threshold_info_e['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 0.04))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_e['threshold'], threshold_info_e['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 0.04))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    plt.show()
    # D
    threshold_info_d, lower, upper = fetch_threshold_info(stock_series, stock_correlation_matrix['D'])
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_d['threshold'], threshold_info_d['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_d['threshold'], threshold_info_d['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    plt.show()
    # T
    threshold_info_t, lower, upper = fetch_threshold_info(stock_series, stock_correlation_matrix['T'])
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_t['threshold'], threshold_info_t['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_t['threshold'], threshold_info_t['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 40, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 40))
    plt.show()


def fetch_stock_n_alr(stock_base_info, N=1):
    # 获取股票在未来一段时间的数据
    date = fetch_future_trade_date(N=N)
    stock_info = pd.DataFrame()
    for code in pd.merge(pro.stock_basic(), stock_base_info, on=['symbol'])['ts_code'].values:
        info = pro.daily(ts_code=code, start_date=date[0], end_date=date[-1])[['ts_code', 'trade_date', 'pct_chg']]
        stock_info = pd.concat([stock_info, info])
    # 计算每只股票的同向比
    data = pd.merge(stock_info, stock_info, on=['trade_date'])
    data['multi'] = data['pct_chg_x'] * data['pct_chg_y']
    data['same_dir'] = data['multi'].apply(lambda x: 1 if x >= 0 else 0)
    data = data.groupby(by=['ts_code_x', 'trade_date']).apply(lambda x: sum(x['same_dir']) / len(x)).reset_index()
    data = data.rename(columns={0: 'same_dir_rate'})
    stock_n_alr = data.groupby(by=['ts_code_x']).apply(lambda x: sum(x['same_dir_rate']) / len(x)).reset_index()
    stock_n_alr = stock_n_alr.rename(columns={0: 'same_dir_rate'})
    stock_n_alr.sort_values(by=['same_dir_rate'], ascending=False, inplace=True)
    stock_n_alr['rank'] = stock_n_alr['same_dir_rate'].rank(ascending=False, method='min')
    return stock_n_alr


def fetch_influence(stock_base_info, data, d=0.85):
    influence = {'DC': [], 'PR': []}
    codes = pd.merge(pro.stock_basic(), stock_base_info, on=['symbol'])['ts_code'].values
    # 度中心性
    rec = defaultdict(float)
    for code1, _, distance in data.values.tolist():
        rec[code1] += 1 / (1 + distance)
    rec = sorted(rec.items(), key=lambda x: x[1], reverse=True)
    for code, dc in rec:
        influence['DC'].append([code, dc])
    influence['DC'] = pd.DataFrame(data=influence['DC'], columns=['code', 'dc'])
    influence['DC']['rank'] = influence['DC']['dc'].rank(ascending=False, method='min')
    # PageRank
    code2num = {code: i for i, code in enumerate(codes)}
    num2code = {i: code for code, i in code2num.items()}
    #
    n = len(codes)
    M = np.zeros((n, n))
    for code1 in codes:
        info = data.loc[data['code1'] == code1]
        size = info.shape[0]
        if size == 0:
            continue
        # PR
        for code2 in info['code2'].values.tolist():
            M[code2num[code2]][code2num[code1]] = 1 / size
    E = np.eye(n)
    I = np.ones((n, 1))
    R = np.dot(np.linalg.inv(E - d * M), (1 - d) / n * I).tolist()
    for num, pr in enumerate(R):
        influence['PR'].append([num2code[num], pr[0]])
    influence['PR'] = pd.DataFrame(data=influence['PR'], columns=['code', 'pr'])
    influence['PR'].sort_values(by=['pr'], ascending=False, inplace=True)
    influence['PR']['rank'] = influence['PR']['pr'].rank(ascending=False, method='min')
    return influence


def fetch_stock_influence(stock_base_info, stock_correlation_matrix, threshold):
    stock_influence = {}
    #
    data = stock_correlation_matrix['E'].loc[stock_correlation_matrix['E']['distance'] <= threshold['E']]
    stock_influence['E'] = fetch_influence(stock_base_info, data)
    #
    data = stock_correlation_matrix['D'].loc[stock_correlation_matrix['D']['distance'] <= threshold['D']]
    stock_influence['D'] = fetch_influence(stock_base_info, data)
    #
    data = stock_correlation_matrix['T'].loc[stock_correlation_matrix['T']['distance'] <= threshold['T']]
    stock_influence['T'] = fetch_influence(stock_base_info, data)
    return stock_influence


def fetch_evaluation(stock_n_alr, stock_influence, N=5):
    exp = set(stock_n_alr.loc[stock_n_alr['rank'] <= N, 'ts_code_x'])
    # E:
    dc_e = stock_influence['E']['DC']
    act_dc_e = set(dc_e.loc[dc_e['rank'] <= N, 'code'].values.tolist())
    pr_e = stock_influence['E']['PR']
    act_pr_e = set(pr_e.loc[pr_e['rank'] <= N, 'code'].values.tolist())
    print('E(DC): %d, E(PR): %d.' % (len(exp & act_dc_e), len(exp & act_pr_e)))
    # D:
    dc_d = stock_influence['D']['DC']
    act_dc_d = set(dc_d.loc[dc_d['rank'] <= N, 'code'].values.tolist())
    pr_d = stock_influence['D']['PR']
    act_pr_d = set(pr_d.loc[pr_d['rank'] <= N, 'code'].values.tolist())
    print('D(DC): %d, D(PR): %d.' % (len(exp & act_dc_d), len(exp & act_pr_d)))
    # T:
    dc_t = stock_influence['T']['DC']
    act_dc_t = set(dc_t.loc[dc_t['rank'] <= N, 'code'].values.tolist())
    pr_t = stock_influence['T']['PR']
    act_pr_t = set(pr_t.loc[pr_t['rank'] <= N, 'code'].values.tolist())
    print('T(DC): %d, T(PR): %d.' % (len(exp & act_dc_t), len(exp & act_pr_t)))


if __name__ == '__main__':
    # Step 1: 获取股票基本信息
    stock_base_info = fetch_stock_base_info()
    # Step 2: 获取股票收益率信息
    stock_price_info = fetch_stock_price_info(stock_base_info)
    # Step 3: 获取不同度量下股票的相关性矩阵
    stock_correlation_matrix = fetch_stock_correlation_matrix(stock_price_info)
    # Step 4: 对不同度量下的结果进行可视化
    # stock_visualization(stock_price_info, stock_correlation_matrix)  # TODO: 寻找合适的示例
    # Step 5: 获取不同度量下对应的阈值
    # fetch_threshold(stock_price_info, stock_correlation_matrix)
    # Step 6: 获取股票的N-ALR指标
    stock_n_alr = fetch_stock_n_alr(stock_base_info, N=1)
    # Step 7: 获取不同网络节点的影响力
    stock_influence = fetch_stock_influence(stock_base_info, stock_correlation_matrix,
                                            threshold={'E': 0.17, 'D': 5.6, 'T': 3.2})
    # Step 8: 影响力评估
    fetch_evaluation(stock_n_alr, stock_influence, N=5)
