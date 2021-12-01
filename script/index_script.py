# Author: LC
# Date Time: 2021/11/20 17:06
# File Description:

from collections import defaultdict
from itertools import combinations_with_replacement

import pandas as pd

from utils.common_function import *
from utils.metric import *

# today = '20211119'


today = datetime.datetime.today().strftime('%Y%m%d')


def is_trading_day():
    return pro.trade_cal(start_date=today, end_date=today)['is_open'].values.tolist()[0] == 1


def fetch_index_price_info():
    index_price_info = pd.DataFrame()
    # 获取时间区间
    trade_date = fetch_trade_date(end_date=today, N=30)
    #
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    try:
        SQL = '''
            SELECT
                B.precursor,
                A.* 
            FROM
                (
                SELECT
                    `code`,
                    DATE_FORMAT( date, '%%Y%%m%%d' ) date,
                    rate pct_chg 
                FROM
                    `discover_index_1d_hist` 
                WHERE
                    exchange = 'HZ' 
                    AND date >= '%s' 
                    AND date <= '%s' 
                ) A
                JOIN ( SELECT * FROM `discover_attributes` ) B ON CAST( A.`code` AS SIGNED ) = B.id 
            ORDER BY
                B.precursor,
                A.`code`,
                A.date
        ''' % (trade_date[0], trade_date[-1])
        cursor.execute(SQL)
        index_price_info = pd.DataFrame(data=cursor.fetchall(), columns=['precursor', 'code', 'date', 'pct_chg'])
    except Exception as e:
        print('Fetch index price info error: %s!' % e)
    finally:
        cursor.close()
        db.close()
        return index_price_info


def fetch_index_price_series(index_price_info):
    return {symbol: info['pct_chg'].values.tolist() for symbol, info in index_price_info.groupby(by=['code'])}


def fetch_influence(index_correlation_matrix):
    influence = []
    # 度中心性
    rec = defaultdict(float)
    for code1, _, distance in index_correlation_matrix.values.tolist():
        rec[code1] += 1 / (1 + distance)
    rec = sorted(rec.items(), key=lambda x: x[1], reverse=True)
    for code, dc in rec:
        influence.append([code, dc])
    influence = pd.DataFrame(data=influence, columns=['code', 'dc'])
    influence['dc'] = influence['dc'] / sum(influence['dc'])
    influence['rank'] = influence['dc'].rank(ascending=False, method='min')
    influence['total'] = len(influence)
    return influence


def fetch_index_influence(index_price_info, index_price_series):
    # 获取不同类别中不同股票间的相关性矩阵
    index_influence = pd.DataFrame()
    columns = ['code1', 'code2', 'distance']
    for precursor, info in index_price_info.groupby(by=['precursor']):
        #
        symbols = sorted(list(set(info['code'])))
        #
        index_correlation_matrix = []
        for code1, code2 in combinations_with_replacement(symbols, 2):
            series1, series2 = index_price_series[code1], index_price_series[code2]
            # tw-lDTW距离
            dist, _ = distance_measure(series1, series2, beta=0.6, punish=0.8, return_path=False)
            index_correlation_matrix.append([code1, code2, dist])
            if code1 != code2:
                dist, _ = distance_measure(series2, series1, beta=0.6, punish=0.8, return_path=False)
                index_correlation_matrix.append([code2, code1, dist])
        index_correlation_matrix = pd.DataFrame(data=index_correlation_matrix, columns=columns)
        influence = fetch_influence(index_correlation_matrix)
        influence['precursor'] = precursor
        index_influence = pd.concat([index_influence, influence])
    return index_influence


def save_index_influence(index_influence):
    data = []
    index_influence['date'] = today
    for code, dc, rank, total, precursor, date in index_influence.values.tolist():
        data.append((precursor, code, dc, int(rank), total, date))
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    try:
        SQL = "REPLACE INTO index_influence_ranking VALUES(%s, %s, %s, %s, %s, %s)"
        cursor.executemany(SQL, data)
        db.commit()
    except Exception as e:
        print('Save index influence error: %s!' % e)
    finally:
        cursor.close()
        db.close()


if __name__ == '__main__':
    # Step 1: 判断当前日期是否为交易日
    if is_trading_day():
        # Step 2: 获取股票收益率信息
        index_price_info = fetch_index_price_info()
        # Step 3: 获取股票收益率序列
        index_price_series = fetch_index_price_series(index_price_info)
        # Step 4: 获取股票影响力
        index_influence = fetch_index_influence(index_price_info, index_price_series)
        # Step 6: 将股票影响力保存至数据库
        save_index_influence(index_influence)
