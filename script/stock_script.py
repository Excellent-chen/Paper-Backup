# Author: LC
# Date Time: 2021/11/19 19:47
# File Description: 股票聚类脚本

from collections import defaultdict
from itertools import combinations_with_replacement

import pandas as pd

from utils.common_function import *
from utils.metric import *

# today = '20211119'


today = datetime.datetime.today().strftime('%Y%m%d')


def is_trading_day():
    return pro.trade_cal(start_date=today, end_date=today)['is_open'].values.tolist()[0] == 1


def fetch_stock_base_info():
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    stock_base_info = pd.DataFrame()
    try:
        SQL = '''
            SELECT
                A.attribute id,
                B.`code` symbol 
            FROM
                ( SELECT * FROM `discover_attribute_belongs` ) A
                JOIN ( SELECT * FROM `discover_stocks` ) B ON A.stock = B.id 
            ORDER BY
                A.attribute,
                B.`code`
        '''
        cursor.execute(SQL)
        stock_base_info = pd.DataFrame(data=cursor.fetchall(), columns=['id', 'symbol'])
    except Exception as e:
        print('Fetch stock base info error: %s!' % e)
    finally:
        cursor.close()
        db.close()
        return stock_base_info


def fetch_stock_price_info():
    # 获取时间区间
    trade_date = fetch_trade_date(end_date=today, N=30)
    #
    stock_price_info = pd.DataFrame()
    for date in trade_date:
        info = pro.daily(trade_date=date)
        stock_price_info = pd.concat([stock_price_info, info])
    stock_price_info.sort_values(by=['ts_code', 'trade_date'], inplace=True)
    stock_price_info['symbol'] = stock_price_info['ts_code'].apply(lambda x: str(x)[:6])
    return stock_price_info


def fetch_stock_price_series(stock_price_info):
    return {symbol: info['pct_chg'].values.tolist() for symbol, info in stock_price_info.groupby(by=['symbol'])}


def fetch_influence(stock_correlation_matrix):
    influence = []
    # 度中心性
    rec = defaultdict(float)
    for code1, _, distance in stock_correlation_matrix.values.tolist():
        rec[code1] += 1 / (1 + distance)
    rec = sorted(rec.items(), key=lambda x: x[1], reverse=True)
    for code, dc in rec:
        influence.append([code, dc])
    influence = pd.DataFrame(data=influence, columns=['code', 'dc'])
    influence['dc'] = influence['dc'] / sum(influence['dc'])
    influence['rank'] = influence['dc'].rank(ascending=False, method='min')
    influence['total'] = len(influence)
    return influence


def fetch_stock_influence(stock_base_info, stock_price_series):
    # 获取不同类别中不同股票间的相关性矩阵
    stock_influence = pd.DataFrame()
    columns = ['code1', 'code2', 'distance']
    for id, info in stock_base_info.groupby(by=['id']):
        #
        symbols = []
        for symbol in info['symbol'].values.tolist():
            if symbol in stock_price_series.keys():
                symbols.append(symbol)
        #
        stock_correlation_matrix = []
        for code1, code2 in combinations_with_replacement(symbols, 2):
            series1, series2 = stock_price_series[code1], stock_price_series[code2]
            # tw-lDTW距离
            dist, _ = distance_measure(series1, series2, beta=0.6, punish=0.8, return_path=False)
            stock_correlation_matrix.append([code1, code2, dist])
            if code1 != code2:
                dist, _ = distance_measure(series2, series1, beta=0.6, punish=0.8, return_path=False)
                stock_correlation_matrix.append([code2, code1, dist])
        stock_correlation_matrix = pd.DataFrame(data=stock_correlation_matrix, columns=columns)
        influence = fetch_influence(stock_correlation_matrix)
        influence['id'] = id
        stock_influence = pd.concat([stock_influence, influence])
    return stock_influence


def save_stock_influence(stock_influence):
    data = []
    stock_influence['date'] = today
    for code, dc, rank, total, id, date in stock_influence.values.tolist():
        data.append((code, dc, int(rank), total, id, date))
    db = get_db(host='haizhiouter.mysql.rds.aliyuncs.com', user='haizhi_root', password='Hai965310', database='stock')
    cursor = db.cursor()
    try:
        SQL = "REPLACE INTO stock_influence_ranking VALUES(%s, %s, %s, %s, %s, %s)"
        cursor.executemany(SQL, data)
        db.commit()
    except Exception as e:
        print('Save stock influence error: %s!' % e)
    finally:
        cursor.close()
        db.close()


if __name__ == '__main__':
    # Step 1: 判断当前日期是否为交易日
    if is_trading_day():
        # Step 2: 获取股票基本信息
        stock_base_info = fetch_stock_base_info()
        # Step 3: 获取股票收益率信息
        stock_price_info = fetch_stock_price_info()
        # Step 4: 获取股票收益率序列
        stock_price_series = fetch_stock_price_series(stock_price_info)
        # Step 5: 获取股票影响力
        stock_influence = fetch_stock_influence(stock_base_info, stock_price_series)
        # Step 6: 将股票影响力保存至数据库
        save_stock_influence(stock_influence)
