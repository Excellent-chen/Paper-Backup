# Author: LC
# Date Time: 2021/11/6 20:18
# File Description:

from collections import defaultdict
from itertools import combinations_with_replacement

from utils.common_function import *
from utils.metric import *


def fetch_index_price_info():
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    index_price_info = pd.DataFrame()
    try:
        SQL = '''
            SELECT
                CONCAT( `code`, '.HZ' ),
                date,
                rate
            FROM
                `discover_index_1d_hist`
            WHERE
                exchange = 'HZ'
                AND `code` >= '000038'
                AND `code` <= '000103' AND date >= '20210301'
                AND date <= '20210901'
        '''
        cursor.execute(SQL)
        index_price_info = pd.DataFrame(data=cursor.fetchall(), columns=['hz_code', 'date', 'pct_chg'])
        index_price_info.sort_values(by=['date', 'hz_code'], inplace=True)
    except Exception as e:
        print('Fetch index price info error: %s!' % e)
    finally:
        cursor.close()
        db.close()
        return index_price_info


def fetch_index_correlation_matrix(index_price_info):
    index_series = {code: info['pct_chg'].values.tolist() for code, info in index_price_info.groupby(by=['hz_code'])}
    index_correlation_matrix = {'E': [], 'D': [], 'T': []}
    for code1, code2 in combinations_with_replacement(index_series.keys(), 2):
        series1, series2 = index_series[code1], index_series[code2]
        # 欧式距离
        if len(series1) == 128 and len(series2) == 128:  # 要求序列长度相等
            dist = np.sqrt(np.sum(np.square(np.array(series1) - np.array(series2)))) / len(series1)
            index_correlation_matrix['E'].append([code1, code2, dist])
            if code1 != code2:
                index_correlation_matrix['E'].append([code2, code1, dist])
        # 常规DTW距离
        dist, _ = distance_measure(series1, series2, beta=0.5, punish=1, return_path=False)
        index_correlation_matrix['D'].append([code1, code2, 2 * dist])
        if code1 != code2:
            index_correlation_matrix['D'].append([code2, code1, 2 * dist])
        # tw-lDTW距离
        dist, _ = distance_measure(series1, series2, beta=0.6, punish=0.8, return_path=False)
        index_correlation_matrix['T'].append([code1, code2, dist])
        if code1 != code2:
            dist, _ = distance_measure(series2, series1, beta=0.6, punish=0.8, return_path=False)
            index_correlation_matrix['T'].append([code2, code1, dist])
    columns = ['code1', 'code2', 'distance']
    index_correlation_matrix['E'] = pd.DataFrame(data=index_correlation_matrix['E'], columns=columns)
    index_correlation_matrix['D'] = pd.DataFrame(data=index_correlation_matrix['D'], columns=columns)
    index_correlation_matrix['T'] = pd.DataFrame(data=index_correlation_matrix['T'], columns=columns)
    return index_correlation_matrix


def index_visualization(index_price_info, index_correlation_matrix, base_code='000038.HZ', index=0):
    index_series = {code: info['pct_chg'].values.tolist() for code, info in index_price_info.groupby(by=['hz_code'])}
    x = [i for i in range(128)]
    y1 = index_series[base_code]
    #
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    # E
    plt.figure(figsize=(10, 2.5))
    matrix_e = index_correlation_matrix['E']
    info_e = matrix_e.loc[(matrix_e['code1'] == base_code) & (matrix_e['code2'] != base_code)]
    info_e.sort_values(by=['distance'], inplace=True)
    y2 = index_series[info_e.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y2, label=info_e.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()
    # D
    plt.figure(figsize=(10, 2.5))
    matrix_d = index_correlation_matrix['D']
    info_d = matrix_d.loc[(matrix_d['code1'] == base_code) & (matrix_d['code2'] != base_code)]
    info_d.sort_values(by=['distance'], inplace=True)
    y3 = index_series[info_d.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y3, label=info_d.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()
    # T
    plt.figure(figsize=(10, 2.5))
    matrix_t = index_correlation_matrix['T']
    info_t = matrix_t.loc[(matrix_t['code1'] == base_code) & (matrix_t['code2'] != base_code)]
    info_t.sort_values(by=['distance'], inplace=True)
    y4 = index_series[info_t.iloc[index, 1]]
    plt.plot(x, y1, label=base_code[:6], linewidth=1, linestyle='solid')
    plt.plot(x, y4, label=info_t.iloc[index, 1][:6], linewidth=1, linestyle='dashed')
    plt.xlabel('相对交易日')
    plt.ylabel('每日收益率')
    plt.legend(loc='upper right')
    plt.show()


def fetch_threshold(index_price_info, index_correlation_matrix):
    index_series = {code: info['pct_chg'].values.tolist() for code, info in index_price_info.groupby(by=['hz_code'])}
    #
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    # E
    threshold_info_e, lower, upper = fetch_threshold_info(index_series, index_correlation_matrix['E'], h=0.01)
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_e['threshold'], threshold_info_e['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 0.04))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_e['threshold'], threshold_info_e['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 0.04))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    plt.show()
    # D
    threshold_info_d, lower, upper = fetch_threshold_info(index_series, index_correlation_matrix['D'])
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_d['threshold'], threshold_info_d['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_d['threshold'], threshold_info_d['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    plt.show()
    # T
    threshold_info_t, lower, upper = fetch_threshold_info(index_series, index_correlation_matrix['T'])
    plt.figure(figsize=(8, 4))
    # 根据阈值与连通子图个数之间的对应关系确定第一个阈值
    plt.subplot(1, 2, 1)
    plt.plot(threshold_info_t['threshold'], threshold_info_t['count'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('连通子图个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    # 根据阈值与最大连通子图节点数之间的对应关系确定第二个阈值
    plt.subplot(1, 2, 2)
    plt.plot(threshold_info_t['threshold'], threshold_info_t['maxsize'])
    plt.xticks(np.arange(lower, upper + 0.01, 1))
    plt.yticks(np.arange(0, 70, 5))
    plt.xlabel('阈值')
    plt.ylabel('最大连通子图节点个数')
    plt.xlim((0, upper))
    plt.ylim((0, 70))
    plt.show()


def fetch_index_n_alr(N=1):
    # 获取指数在未来一段时间的数据
    date = fetch_future_trade_date(N=N)
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    index_n_alr = pd.DataFrame()
    try:
        SQL = '''
            SELECT
                CONCAT( `code`, '.HZ' ),
                date,
                rate
            FROM
                `discover_index_1d_hist`
            WHERE
                exchange = 'HZ'
                AND `code` >= '000038'
                AND `code` <= '000103' AND date >= '%s'
                AND date <= '%s'
        ''' % (date[0], date[-1])
        cursor.execute(SQL)
        index_info = pd.DataFrame(data=cursor.fetchall(), columns=['hz_code', 'date', 'pct_chg'])
        index_info.sort_values(by=['date', 'hz_code'], inplace=True)
    except Exception as e:
        print('Fetch index info error: %s!' % e)
    else:
        data = pd.merge(index_info, index_info, on=['date'])
        data['multi'] = data['pct_chg_x'] * data['pct_chg_y']
        data['same_dir'] = data['multi'].apply(lambda x: 1 if x >= 0 else 0)
        data = data.groupby(by=['hz_code_x', 'date']).apply(lambda x: sum(x['same_dir']) / len(x)).reset_index()
        data = data.rename(columns={0: 'same_dir_rate'})
        index_n_alr = data.groupby(by=['hz_code_x']).apply(lambda x: sum(x['same_dir_rate']) / len(x)).reset_index()
        index_n_alr = index_n_alr.rename(columns={0: 'same_dir_rate'})
        index_n_alr.sort_values(by=['same_dir_rate'], ascending=False, inplace=True)
        index_n_alr['rank'] = index_n_alr['same_dir_rate'].rank(ascending=False, method='min')
    finally:
        cursor.close()
        db.close()
        return index_n_alr


def fetch_influence(codes, data, d=0.85):
    influence = {'DC': [], 'PR': []}
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


def fetch_index_influence(index_price_info, index_correlation_matrix, threshold):
    index_influence = {}
    codes = sorted(list(set(index_price_info['hz_code'])))
    #
    data = index_correlation_matrix['E'].loc[index_correlation_matrix['E']['distance'] <= threshold['E']]
    index_influence['E'] = fetch_influence(codes, data)
    #
    data = index_correlation_matrix['D'].loc[index_correlation_matrix['D']['distance'] <= threshold['D']]
    index_influence['D'] = fetch_influence(codes, data)
    #
    data = index_correlation_matrix['T'].loc[index_correlation_matrix['T']['distance'] <= threshold['T']]
    index_influence['T'] = fetch_influence(codes, data)
    return index_influence


def fetch_evaluation(index_influence, index_n_alr, N=5):
    exp = set(index_n_alr.loc[index_n_alr['rank'] <= N, 'hz_code_x'])
    # E:
    dc_e = index_influence['E']['DC']
    act_dc_e = set(dc_e.loc[dc_e['rank'] <= N, 'code'].values.tolist())
    pr_e = index_influence['E']['PR']
    act_pr_e = set(pr_e.loc[pr_e['rank'] <= N, 'code'].values.tolist())
    print('E(DC): %d, E(PR): %d.' % (len(exp & act_dc_e), len(exp & act_pr_e)))
    # D:
    dc_d = index_influence['D']['DC']
    act_dc_d = set(dc_d.loc[dc_d['rank'] <= N, 'code'].values.tolist())
    pr_d = index_influence['D']['PR']
    act_pr_d = set(pr_d.loc[pr_d['rank'] <= N, 'code'].values.tolist())
    print('D(DC): %d, D(PR): %d.' % (len(exp & act_dc_d), len(exp & act_pr_d)))
    # T:
    dc_t = index_influence['T']['DC']
    act_dc_t = set(dc_t.loc[dc_t['rank'] <= N, 'code'].values.tolist())
    pr_t = index_influence['T']['PR']
    act_pr_t = set(pr_t.loc[pr_t['rank'] <= N, 'code'].values.tolist())
    print('T(DC): %d, T(PR): %d.' % (len(exp & act_dc_t), len(exp & act_pr_t)))


if __name__ == '__main__':
    # Step 1: 获取指数收益率数据
    index_price_info = fetch_index_price_info()
    # Step 2: 获取不同度量下指数的相关性矩阵
    index_correlation_matrix = fetch_index_correlation_matrix(index_price_info)
    # Step 3: 对不同度量下的结果进行可视化
    # index_visualization(index_price_info, index_correlation_matrix)  # TODO: 寻找合适的示例
    # Step 4: 获取不同度量下对应的阈值
    # fetch_threshold(index_price_info, index_correlation_matrix)
    # Step 5: 获取指数的N-ALR指标
    index_n_alr = fetch_index_n_alr(N=1)
    # Step 6: 获取不同网络节点的影响力
    index_influence = fetch_index_influence(index_price_info, index_correlation_matrix,
                                            threshold={'E': 0.14, 'D': 2.0, 'T': 1.1})
    # Step 7: 影响力评估
    fetch_evaluation(index_influence, index_n_alr, N=5)
