# Author: LC
# Date Time: 2021/11/5 14:15
# File Description:

import pymysql
import tushare as ts

pro = ts.pro_api('5e440fc23c7094ffebec94e06607adaf3a47cb337c6aeb63ba5fad71')


# 连接数据库
def get_db(host="10.249.42.85", user="root", password="965310", database="chen"):  # 内网
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset="utf8", use_unicode=True)
    return db


# 获取历史交易日期
def fetch_trade_date(N=120):
    trade_date = sorted(
        pro.trade_cal(start_date='20210101', end_date='20210901', is_open='1')['cal_date'].values.tolist())
    return trade_date[-N:]


# 获取未来交易日期
def fetch_future_trade_date(start_date='20210902', N=1):
    future_trade_date = pro.trade_cal(start_date=start_date, is_open='1')
    future_trade_date.sort_values(by=['cal_date'], inplace=True)
    return future_trade_date['cal_date'].values.tolist()[:N]
