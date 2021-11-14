# Author: LC
# Date Time: 2021/11/13 19:34
# File Description: 双加权收益率

import datetime


# 计算 a 的股数
def get_volumes_a(trade_logs_a, prices_a):
    volumes_a = []
    for i in range(len(trade_logs_a)):
        if i == 0:
            volumes_a.append(trade_logs_a[i] / prices_a[i])
        else:
            volumes_a.append(volumes_a[i - 1] + trade_logs_a[i] / prices_a[i])
    return volumes_a


# 计算 a 的涨幅
def get_rates_a(prices_a):
    rates_a = []
    for i in range(1, len(prices_a)):
        rates_a.append(prices_a[i] / prices_a[i - 1] - 1)
    return rates_a


# 计算 e 的涨幅
def get_rates_e(trade_logs_a, funds_all, prices_a, volumes_a):
    funds_a, funds_e = [], []
    for i in range(len(funds_all)):
        funds_a.append(prices_a[i] * volumes_a[i])
        funds_e.append(funds_all[i] - funds_a[i])
    rates_e = []
    for i in range(1, len(funds_all)):
        rates_e.append((funds_e[i] + trade_logs_a[i]) / funds_e[i - 1] - 1)
    return rates_e


# 获取简单战斗
def get_simple_battles(trade_logs_a, prices_a):
    simple_battles = []
    volumes_changes_a = []  # a 的股数变动
    for i in range(len(trade_logs_a)):
        volumes_changes_a.append(trade_logs_a[i] / prices_a[i])
    for i in range(len(volumes_changes_a)):
        for j in range(len(simple_battles)):
            simple_battles[j][i] = simple_battles[j][i - 1]
        if volumes_changes_a[i] == 0:
            continue
        elif volumes_changes_a[i] > 0:  # 买入
            battle = [0] * len(volumes_changes_a)
            battle[i] = volumes_changes_a[i]
            simple_battles.append(battle)
        else:  # 卖出
            sell_volume = -volumes_changes_a[i]
            battle_id = len(simple_battles) - 1
            while battle_id >= 0 and sell_volume > 0.00001:
                cur_battle = simple_battles[battle_id]
                if sell_volume >= cur_battle[i]:
                    sell_volume -= cur_battle[i]
                    cur_battle[i] = 0
                else:
                    hold_battle = [0] * len(volumes_changes_a)
                    close_battle = [0] * len(volumes_changes_a)
                    for k in range(i + 1):
                        close_battle[k] = cur_battle[k] * sell_volume / cur_battle[i]
                        hold_battle[k] = cur_battle[k] - close_battle[k]
                    close_battle[i] = 0
                    del simple_battles[battle_id]
                    simple_battles.insert(battle_id, close_battle)
                    simple_battles.insert(battle_id, hold_battle)
                    sell_volume = 0
                battle_id -= 1
    return simple_battles


# 获取网格及市值
def get_grids_and_funds(trade_logs_a, funds_all, rates_a, rates_e):  # 获取网格及其市值
    init_fund = funds_all[0]
    funds_grids, types_grids = [], []
    # 按天遍历
    for day in range(len(funds_all)):
        # 上一天各网格的市值折算到今天
        if day == 0:
            funds_grids.append([init_fund])
            types_grids.append(['e'])
        else:
            for idx_grid in range(len(funds_grids)):
                length = len(funds_grids[idx_grid])
                if types_grids[idx_grid][length - 1] == 'a':
                    funds_grids[idx_grid].append(funds_grids[idx_grid][length - 1] * (1 + rates_a[length - 1]))
                    types_grids[idx_grid].append('a')
                else:
                    funds_grids[idx_grid].append(funds_grids[idx_grid][length - 1] * (1 + rates_e[length - 1]))
                    types_grids[idx_grid].append('e')
        #
        cur_trade_a = trade_logs_a[day]
        if abs(cur_trade_a) < 0.00001:
            continue
        elif cur_trade_a > 0.00001:  # 买入
            # 从下往上遍历网格
            idx_grid = 0
            while idx_grid < len(funds_grids) and cur_trade_a > 0.00001:
                cur_types = types_grids[idx_grid]
                cur_funds = funds_grids[idx_grid]
                if cur_types[day] == 'e':
                    if cur_trade_a >= cur_funds[day]:
                        cur_types[day] = 'a'
                        cur_trade_a -= cur_funds[day]
                    else:
                        a_fund_grid = [0] * len(cur_funds)
                        e_fund_grid = [0] * len(cur_funds)
                        for tmp_idx in range(day + 1):
                            a_fund_grid[tmp_idx] = cur_funds[tmp_idx] * cur_trade_a / cur_funds[day]
                            e_fund_grid[tmp_idx] = cur_funds[tmp_idx] - a_fund_grid[tmp_idx]
                        a_type_grid = cur_types.copy()
                        e_type_grid = cur_types.copy()
                        a_type_grid[day] = 'a'
                        del funds_grids[idx_grid]
                        funds_grids.insert(idx_grid, e_fund_grid)
                        funds_grids.insert(idx_grid, a_fund_grid)
                        del types_grids[idx_grid]
                        types_grids.insert(idx_grid, e_type_grid)
                        types_grids.insert(idx_grid, a_type_grid)
                        cur_trade_a = 0
                idx_grid += 1
        else:  # 卖出
            cur_trade_a = -cur_trade_a
            idx_grid = len(funds_grids) - 1
            while idx_grid >= 0 and cur_trade_a > 0.00001:
                cur_types = types_grids[idx_grid]
                cur_funds = funds_grids[idx_grid]
                if cur_types[day] == 'a':
                    if cur_trade_a - cur_funds[day] > 0.00001 or abs(cur_trade_a - cur_funds[day]) < 0.00001:
                        cur_types[day] = 'e'
                        cur_trade_a -= cur_funds[day]
                    else:
                        a_fund_grid = [0] * len(cur_funds)
                        e_fund_grid = [0] * len(cur_funds)
                        for tmp_idx in range(day + 1):
                            e_fund_grid[tmp_idx] = cur_funds[tmp_idx] * cur_trade_a / cur_funds[day]
                            a_fund_grid[tmp_idx] = cur_funds[tmp_idx] - e_fund_grid[tmp_idx]
                        a_type_grid = cur_types.copy()
                        e_type_grid = cur_types.copy()
                        e_type_grid[day] = 'e'
                        del funds_grids[idx_grid]
                        funds_grids.insert(idx_grid, e_fund_grid)
                        funds_grids.insert(idx_grid, a_fund_grid)
                        del types_grids[idx_grid]
                        types_grids.insert(idx_grid, e_type_grid)
                        types_grids.insert(idx_grid, a_type_grid)
                idx_grid -= 1
    return funds_grids, types_grids


# 获取各个简单战斗的初始成本
def get_init_costs(simple_battles, rates_a, rates_e, prices_a):  # 计算简单战斗的初始成本
    init_costs = [0] * len(simple_battles)
    # 逐个计算简单战斗的初始成本
    fund_simple_battles = []
    for battle_idx in range(len(simple_battles)):
        cur_battle = simple_battles[battle_idx]
        fund_simple_battles.append(simple_battles[battle_idx].copy())
        for tmp_day in range(1, len(fund_simple_battles[battle_idx])):
            fund_simple_battles[battle_idx][tmp_day] = simple_battles[battle_idx][tmp_day - 1] * prices_a[tmp_day]
        start_day = 0
        while start_day < len(cur_battle) and cur_battle[start_day] == 0:
            start_day += 1
        if start_day == len(cur_battle):
            break
        cur_init_cost = cur_battle[start_day] * prices_a[start_day]
        fund_simple_battles[battle_idx][start_day] = cur_init_cost
        exclude_battle_idxs = []
        for tmp_idx in range(battle_idx):
            if simple_battles[tmp_idx][start_day] > 0.00001:
                exclude_battle_idxs.append(tmp_idx)
        for day in range(start_day - 1, -1, -1):
            exclude_fund_cur_day = 0
            for key, battle in enumerate(fund_simple_battles):
                if key in exclude_battle_idxs:
                    exclude_fund_cur_day += battle[day + 1]
            fund_a_cur_day = 0
            for key, battle in enumerate(simple_battles):
                fund_a_cur_day += battle[day] * prices_a[day + 1]
            include_fund_a = (fund_a_cur_day - exclude_fund_cur_day) if fund_a_cur_day > exclude_fund_cur_day else 0
            if include_fund_a >= cur_init_cost:
                cur_init_cost = cur_init_cost / (rates_a[day] + 1)
            else:
                cur_init_cost = include_fund_a / (rates_e[day] + 1) + (cur_init_cost - include_fund_a) / (
                        rates_e[day] + 1)
            fund_simple_battles[battle_idx][day] = cur_init_cost
        init_costs[battle_idx] = cur_init_cost
    return init_costs


def diff_days(start_day, end_day):
    start_day = datetime.datetime.strptime(start_day, '%Y-%m-%d').date()
    end_day = datetime.datetime.strptime(end_day, '%Y-%m-%d').date()
    return (end_day - start_day).days


def get_profit_of_simple_battle(simple_battle, prices_a):
    ans, volume = 0, 0
    for day, val in enumerate(simple_battle):
        if volume == 0 and val > 0.00001:
            ans += -val * prices_a[day]
            volume = val
        if volume > 0.00001 and val == 0:
            ans += volume * prices_a[day]
            break
    return ans


def get_days_of_simple_battle(simple_battle, dates):
    volume = 0
    start_day, end_day = None, None
    for day, val in enumerate(simple_battle):
        if volume == 0 and val > 0.00001:
            volume = val
            start_day = dates[day]
        if volume > 0.00001 and val == 0:
            end_day = dates[day]
            break
    if not start_day or not end_day:
        return 0
    else:
        return diff_days(start_day, end_day)


def get_wwr(simple_battles, prices_a, init_costs, dates):
    total_day = diff_days(dates[0], dates[-1])
    if total_day == 0:
        return 0
    else:
        profits = []
        hold_days = []
        for idx, simple_battle in enumerate(simple_battles):
            profits.append(get_profit_of_simple_battle(simple_battle, prices_a))
            hold_days.append(get_days_of_simple_battle(simple_battle, dates))
        wwcs = []  # 各简单战斗的双加权成本
        for i in range(len(init_costs)):
            if hold_days[i] == 0:
                wwcs.append(0)
            else:
                wwcs.append(init_costs[i] * hold_days[i] / total_day)
        total_profit, total_wwc = sum(profits), sum(wwcs)
        if total_wwc == 0:
            print('总双加权成本为零')
            return 0
        else:
            print(total_profit, total_wwc)
            return total_profit / total_wwc


def wwr_main(dates, funds_all, trade_logs_a, prices_a):
    #
    volumes_a = get_volumes_a(trade_logs_a, prices_a)
    #
    rates_a = get_rates_a(prices_a)
    #
    rates_e = get_rates_e(trade_logs_a, funds_all, prices_a, volumes_a)
    #
    simple_battles = get_simple_battles(trade_logs_a, prices_a)
    #
    funds_grids, types_grids = get_grids_and_funds(trade_logs_a, funds_all, rates_a, rates_e)
    #
    init_costs = get_init_costs(simple_battles, rates_a, rates_e, prices_a)
    #
    wwr_val = get_wwr(simple_battles, prices_a, init_costs, dates)
    #
    return wwr_val


def wwr_main_per_day(dates, funds_all, trade_logs_a, prices_a):
    volumes_a = get_volumes_a(trade_logs_a, prices_a)

    wwrs_per_day = []

    cur_dates = []
    cur_funds_all = []
    cur_prices_a = []

    for i in range(len(dates)):
        cur_volume_a = volumes_a[i]

        cur_dates.append(dates[i])
        cur_funds_all.append(funds_all[i])
        cur_prices_a.append(prices_a[i])

        cur_trade_logs_a = []
        for j in range(i):
            cur_trade_logs_a.append(trade_logs_a[j])
        cur_trade_logs_a.append(trade_logs_a[i] - cur_volume_a * prices_a[i])  # 当前持仓全部卖掉

        wwrs_per_day.append(wwr_main(cur_dates, cur_funds_all, cur_trade_logs_a, cur_prices_a))

    return wwrs_per_day


if __name__ == '__main__':
    # datas = [
    #     [
    #         ["2015-01-01", "2015-04-01", "2015-06-01", "2015-11-01", "2016-01-01"],
    #         [3000000, 3500000, 2900000, 2800000, 3100000],
    #         [1000000, 1500000, -1800000, 900000, -1500000],
    #         [10, 12, 10.22, 14.31, 13.42]
    #     ]
    # ]
    datas = [
        [
            ["2015-01-01", "2015-06-01", "2015-10-01", "2015-11-01", "2016-01-01"],
            [300000, 500000, 270000, 280000, 290000],
            [100000, 200000, -120000, 48000, -168000],
            [10, 20, 10, 12, 14]
        ]
    ]

    dates, funds_all, trade_logs_a, prices_a = datas[0]  # 日期，个人总资产，股票资金流动，股票单价

    wwr_val_per_day = wwr_main_per_day(dates, funds_all, trade_logs_a, prices_a)
