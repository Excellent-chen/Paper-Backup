# Author: LC
# Date Time: 2021/11/5 14:13
# File Description:

from utils.common_function import *
from utils.evaluation import *


def fetch_stock_class_info():
    db = get_db()
    cursor = db.cursor()
    stock_class_info = pd.DataFrame()
    try:
        SQL = '''
            SELECT
                B.precursor,
                A.id,
                A.`code` 
            FROM
                ( SELECT * FROM `semantic_attribute_contains_history` WHERE date = '2021-02-23' ) A
                JOIN ( SELECT * FROM `semantic_attributes_history` WHERE date = '2021-02-23' ) B ON A.id = B.id
        '''
        cursor.execute(SQL)
        stock_class_info = pd.DataFrame(data=cursor.fetchall(), columns=['precursor', 'id', 'symbol'])
        stock_class_info = pd.merge(stock_class_info, pro.stock_basic(), on=['symbol'])[['precursor', 'id', 'ts_code']]
        stock_class_info.sort_values(by=['precursor', 'id', 'ts_code'], inplace=True)
    except Exception as e:
        print('Fetch stock base info error: %s!' % e)
    finally:
        cursor.close()
        db.close()
        return stock_class_info


def fetch_stock_price_info():
    stock_price_info = pd.DataFrame()
    for date in fetch_trade_date():
        info = pro.daily(trade_date=date)
        stock_price_info = pd.concat([stock_price_info, info])
    stock_price_info.sort_values(by=['trade_date', 'ts_code'], inplace=True)
    return stock_price_info


def class_evaluation(stock_class_info, stock_price_info):
    for start_date, N in (('20210722', 30), ('20210609', 60), ('20210423', 90), ('20210311', 120)):
        price_info = stock_price_info.loc[stock_price_info['trade_date'] >= start_date]
        for precursor, info in stock_class_info.groupby(by=['precursor']):
            data = pd.merge(info, price_info, on=['ts_code'])
            data.sort_values(by=['precursor', 'id', 'ts_code', 'trade_date'])
            # Precursor 1: 行业. 2: 地域. 3: 概念.
            asdrc, ardc = fetch_asdrc(data), fetch_ardc(data)
            sdrdc = fetch_sdrdc(asdrc, ardc)
            print('N: %d. Precursor: %d. ASDRC: %.4f. ARDC: %.4f. SDRDC:%.4f' % (N, precursor, asdrc, ardc, sdrdc))


def visualization():
    #
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    #
    df = pro.index_daily(ts_code='000001.SH', start_date='20210311', end_date='20210901')  # 上证指数
    df.sort_values(by=['trade_date'], inplace=True)
    x = df['trade_date'].values.tolist()
    plt.figure(figsize=(5, 3.5))
    plt.plot(x, df['close'].values.tolist(), linewidth='1')
    ax = plt.gca()
    xml = ticker.MultipleLocator(len(x) // 8)
    ax.xaxis.set_major_locator(xml)
    plt.xticks(rotation=30)
    plt.xlabel('日期')
    plt.ylabel('收益指数值')
    plt.show()


if __name__ == '__main__':
    # Step 1: 获取同花顺分类体系
    stock_class_info = fetch_stock_class_info()
    # Step 2:
    stock_price_info = fetch_stock_price_info()
    #
    class_evaluation(stock_class_info, stock_price_info)
    #
    visualization()
