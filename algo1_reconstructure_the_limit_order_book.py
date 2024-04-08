import warnings
import pandas as pd
import numpy as np
import datetime
import os
import glob

root_path = r"./"

# 在这里更改处理的股票
stock_index = ['000963', '000958', '002014', '002139', '002243', '002369', '002479', '002752', '002881']

# 生成文件存储路径
# os.makedirs(r'./dfs/'+stock_index, exist_ok=True)

# ------------------------------------数据预处理----------------------------------------------


def read_df_items(path, this_cols, this_cols_name):
    file = glob.glob(os.path.join(path, "*.csv"))
    dl = []
    for f in file:
        dl.append(pd.read_csv(f, usecols=this_cols, names=this_cols_name, header=0))
    return dl


def read_df(stock, table, this_cols, this_cols_name):
    this_path = r'./实证数据/lob/2019/' + stock + '/am_' + table
    dl_1 = read_df_items(this_path, this_cols, this_cols_name)
    this_path = r'./实证数据/lob/2019/' + stock + '/pm_' + table
    dl_2 = read_df_items(this_path, this_cols, this_cols_name)

    if len(dl_1) != len(dl_2):
        warnings.warn('数据日期出错')

    dl = []

    for j in range(len(dl_1)):
        this_sheet = pd.concat([dl_1[j], dl_2[j]], axis=0)
        this_sheet = this_sheet.copy().reset_index(drop=True)
        dl.append(this_sheet)

    return dl


# --------------------------------------def------------------------------------------------


def make_deal(this_price, this_volume, this_level, this_direction):
    if this_direction == 1:
        deal_level = this_level.loc[this_level['ask_p'] <= this_price].copy()
        deal_level['v_sum_1'] = deal_level['ask_v'].cumsum()
        for index, row in deal_level.iterrows():
            if this_volume >= row['v_sum_1']:
                this_level.loc[this_level['ask_p'] == row['ask_p'], 'ask_v'] = 0
            elif this_volume < row['v_sum_1']:
                this_level.loc[this_level['ask_p'] == row['ask_p'], 'ask_v'] = row['v_sum_1'] - this_volume
                break
        count_zero = len(this_level[this_level['ask_v'] == 0])
        this_level[['ask_p', 'ask_v']] = this_level[['ask_p', 'ask_v']].shift(-count_zero).copy()
        if this_volume > deal_level['v_sum_1'].iloc[-1]:
            if not np.isnan(this_level['bid_p'].iloc[-1]):
                this_level.loc[len(this_level.index)+1] = [np.nan] * 4
            this_level[['bid_p', 'bid_v']] = this_level[['bid_p', 'bid_v']].shift(1).copy()
            this_level['bid_p'][1] = this_price
            this_level['bid_v'][1] = this_volume - deal_level['v_sum_1'].iloc[-1]

    elif this_direction == 2:
        deal_level = this_level.loc[this_level['bid_p'] >= this_price].copy()
        deal_level['v_sum_2'] = deal_level['bid_v'].cumsum()
        for index, row in deal_level.iterrows():
            if this_volume >= row['v_sum_2']:
                this_level.loc[this_level['bid_p'] == row['bid_p'], 'bid_v'] = 0
            elif this_volume < row['v_sum_2']:
                this_level.loc[this_level['bid_p'] == row['bid_p'], 'bid_v'] = row['v_sum_2'] - this_volume
                break
        count_zero = len(this_level[this_level['bid_v'] == 0])
        this_level[['bid_p', 'bid_v']] = this_level[['bid_p', 'bid_v']].shift(-count_zero).copy()
        if this_volume > deal_level['v_sum_2'].iloc[-1]:
            if not np.isnan(this_level['ask_p'].iloc[-1]):
                this_level.loc[len(this_level.index) + 1] = [np.nan] * 4
            this_level[['ask_p', 'ask_v']] = this_level[['ask_p', 'ask_v']].shift(1).copy()
            this_level['ask_p'][1] = this_price
            this_level['ask_v'][1] = this_volume - deal_level['v_sum_2'].iloc[-1]
    return this_level


def gen_tail(this_level):
    this_tail = np.array(this_level.loc[1:10].copy())
    this_tail = np.reshape(this_tail.copy(), [1, -1]).tolist()
    this_tail = this_tail[0]
    return this_tail


def get_rv(this_window_snap, gap_num):
    this_window_snap['mid_price'] = (this_window_snap['AskP1'] + this_window_snap['BidP1']) / 2
    index_list = []
    for this_index in range(this_window_snap.index[0], this_window_snap.index[-1] + 1, int(gap_num/3)):  # 隔行取数据
        index_list.append(this_index)
    this_window_snap = this_window_snap.loc[index_list]
    this_window_snap["mid_lag"] = this_window_snap["mid_price"].shift(1)
    this_window_snap["ln_mid"] = this_window_snap["mid_price"].apply(np.log)
    this_window_snap["ln_mid_lag"] = this_window_snap["mid_lag"].apply(np.log)
    this_window_snap["r"] = this_window_snap["ln_mid"] - this_window_snap["ln_mid_lag"]
    # this_window_snap["r^2"] = this_window_snap["r"] * this_window_snap["r"]

    return this_window_snap['r']


def main(stock_index = '000516'):
    # 逐笔成交表 Trade
    trade_spot_col = [0, 1, 10, 11, 12, 13, 14]  # 列数从0开始
    trade_spot_col = [x + 1 for x in trade_spot_col]
    trade_spot_col_name = ['Tradedate', 'OrigTime', 'BidApplSeqNum', 'AskApplSeqNum', 'Price', 'TradeQty', 'ExecType']
    trade_spot_dl = read_df(stock_index, 'hq_trade_spot', trade_spot_col, trade_spot_col_name)

    # 逐笔委托行情表 order
    order_spot_col = [0, 1, 7, 10, 11, 13]
    order_spot_col = [x + 1 for x in order_spot_col]
    order_spot_col_name = ['Tradedate', 'OrigTime', 'ApplSeqNum', 'Price', 'TradeQty', 'ExecType']
    order_spot_dl = read_df(stock_index, 'hq_order_spot', order_spot_col, order_spot_col_name)

    # 证券行情快照档位表 snap_level
    snap_level_spot_col = [0, 1] + [x for x in range(9, 49)]
    snap_level_spot_col = [x + 1 for x in snap_level_spot_col]
    level_name = []
    for i in range(1, 11):
        level_name = level_name + ['AskP' + str(i), 'BidP' + str(i), 'AskV' + str(i), 'BidV' + str(i)]
    snap_level_spot_col_name = ['Tradedate', 'OrigTime'] + level_name
    snap_level_spot_dl = read_df(stock_index, 'snap_level_spot', snap_level_spot_col, snap_level_spot_col_name)

    # ------------------------------------lob复现----------------------------------------------

    this_path = r'./实证数据/lob/2019/' + stock_index + '/am_hq_order_spot'
    file_date = glob.glob(os.path.join(this_path, "*.csv"))
    file_date = [x[-8:-4] for x in file_date]

    if len(file_date) != len(trade_spot_dl):
        warnings.warn('日期匹配错误')

    for k in range(len(trade_spot_dl)):
        snap_level_spot = snap_level_spot_dl[k]
        order_spot = order_spot_dl[k]
        trade_spot = trade_spot_dl[k]
        file_name = file_date[k]

        # 第一笔订单之前的snap记录无效，为了方便校对时的循环，删掉
        snap_level_spot = snap_level_spot.loc[snap_level_spot['OrigTime'] > order_spot.iloc[0, 1]].copy()

        trade_spot['OrigTime'] = trade_spot['OrigTime'] - trade_spot['Tradedate'] * 1000000000
        snap_level_spot['OrigTime'] = snap_level_spot['OrigTime'] - snap_level_spot['Tradedate'] * 1000000000
        order_spot['OrigTime'] = order_spot['OrigTime'] - order_spot['Tradedate'] * 1000000000

        # 源数据里面，撤单委托不显示价格，先加上，方便后续处理
        cancel_trade = trade_spot[trade_spot['Price'] == 0]  # 前提：撤单的价格全为0， 不存在为0的正常订单
        cancel_trade['SeqNum'] = cancel_trade['BidApplSeqNum'].values + cancel_trade['AskApplSeqNum'].values
        tmp_trade = pd.merge(cancel_trade['SeqNum'], order_spot[['ApplSeqNum', 'Price']], left_on='SeqNum',
                             right_on=['ApplSeqNum'], how='left')
        tmp_trade.index = cancel_trade.index
        trade_spot.loc[trade_spot['Price'] == 0, 'Price'] = tmp_trade['Price']
        del tmp_trade, cancel_trade

        order_spot['SheetType'] = 0
        trade_spot['SheetType'] = 1

        tmp_df = pd.concat([trade_spot, order_spot], axis=0)
        tmp_df = tmp_df.copy().sort_values(by=['Tradedate', 'OrigTime', 'SheetType'], axis=0)
        # tmp_df = tmp_df[tmp_df['OrigTime'] >= 93000000]

        # current_level = snap_level_spot[snap_level_spot['OrigTime'] == 93000000]
        # current_level = np.array(current_level.copy().iloc[:, 2:])
        # current_level = pd.DataFrame(np.reshape(current_level, [10, 4]))
        # current_level.index = [x for x in range(1, 11)]
        # current_level.columns = ['ask_p', 'bid_p', 'ask_v', 'bid_v']
        current_level = pd.DataFrame(columns=['ask_p', 'bid_p', 'ask_v', 'bid_v'], index=[x for x in range(1, 11)])

        flag_925 = True
        correction = False
        time_stamp = len(snap_level_spot[snap_level_spot['OrigTime'] < 92500000].index) + 1

        snap_time = 92500000
        last_time = order_spot.iloc[0, 1]

        level_tail = [0] * 40
        initiate_side = 0
        # initiate_side:0→集合竞价; 1→买方bid发起; 2→卖方ask发起
        output_df = pd.DataFrame(
            columns=['time', 'price', 'volume', 't_bid_num', 't_ask_num', 'initiate_side'] + level_name)

        # 日内行遍历
        for time, sheet, price, volume, direction, order_num, t_bid_num, t_ask_num in list(
                zip(tmp_df['OrigTime'], tmp_df['SheetType'], tmp_df['Price'], tmp_df['TradeQty'], tmp_df['ExecType'],
                    tmp_df['ApplSeqNum'], tmp_df['BidApplSeqNum'], tmp_df['AskApplSeqNum'])):

            # 与snap校对
            if 93000000 < time < 145700000 and last_time <= snap_time < time:
                if level_tail != snap_level_spot.iloc[time_stamp, 2:].to_list():
                    # print('====================出现配对错误！==========================')
                    # print('时间:' + str(last_time))
                    # print('计算挡位' + str(level_tail))
                    # print('实际挡位' + str(snap_level_spot.iloc[time_stamp, 2:].to_list()))
                    if correction:
                        current_level.iloc[0:10] = pd.DataFrame(
                            np.reshape(np.array(snap_level_spot.iloc[time_stamp, 2:]), [-1, 4]))
                        correction = False
                    else:
                        correction = True
                # else:
                # print('======================配对无误==============================')
                # print('时间:' + str(last_time))
                # print('计算挡位' + str(level_tail))
                # print('实际挡位' + str(snap_level_spot.iloc[time_stamp, 2:].to_list()))

                time_stamp = time_stamp + 1
                snap_time = snap_level_spot.iloc[time_stamp, 1]
            while last_time > snap_time:
                time_stamp = time_stamp + 1
                snap_time = snap_level_spot.iloc[time_stamp, 1]
            last_time = time

            if sheet == 0:
                # current_level = add_to_snap(price, volume, current_level, direction, order_num)
                if direction == 1:  # 买方
                    if price >= current_level['ask_p'][1]:
                        if time < 93000000:
                            # 集合竞价，不成交
                            if price in current_level['bid_p'].values:
                                # 本来就有这个挡位时
                                current_level.loc[current_level['bid_p'] == price, 'bid_v'] = current_level.loc[
                                                                                                  current_level[
                                                                                                      'bid_p'] == price, 'bid_v'] + volume
                            else:
                                # 新增一个挡位
                                current_level.loc[len(current_level.index) + 1] = [np.nan, price, np.nan, volume]
                                tmp_level = current_level[['bid_p', 'bid_v']].copy()
                                tmp_level = tmp_level.sort_values(by='bid_p', ascending=False)
                                tmp_level = tmp_level.reset_index(drop=True)
                                tmp_level.index = tmp_level.index + 1
                                current_level[['bid_p', 'bid_v']] = tmp_level.copy()
                        else:
                            current_level = make_deal(price, volume, current_level, 1)
                            initiate_side = 1
                    else:
                        if price in current_level['bid_p'].values:
                            # 本来就有这个挡位时
                            current_level.loc[current_level['bid_p'] == price, 'bid_v'] = current_level.loc[
                                                                                              current_level[
                                                                                                  'bid_p'] == price, 'bid_v'] + volume
                        else:
                            # 新增一个挡位
                            current_level.loc[len(current_level.index) + 1] = [np.nan, price, np.nan, volume]
                            tmp_level = current_level[['bid_p', 'bid_v']].copy()
                            tmp_level = tmp_level.sort_values(by='bid_p', ascending=False)
                            tmp_level = tmp_level.reset_index(drop=True)
                            tmp_level.index = tmp_level.index + 1
                            current_level[['bid_p', 'bid_v']] = tmp_level.copy()
                elif direction == 2:
                    if price <= current_level['bid_p'][1]:
                        if time < 93000000:
                            # 集合竞价，不成交
                            if price in current_level['ask_p'].values:
                                # 本来就有这个挡位时
                                current_level.loc[current_level['ask_p'] == price, 'ask_v'] = current_level.loc[
                                                                                                  current_level[
                                                                                                      'ask_p'] == price, 'ask_v'] + volume
                            else:
                                # 新增一个挡位
                                current_level.loc[len(current_level.index) + 1] = [price, np.nan, volume, np.nan]
                                tmp_level = current_level[['ask_p', 'ask_v']].copy()
                                tmp_level = tmp_level.sort_values(by='ask_p', ascending=True)
                                tmp_level = tmp_level.reset_index(drop=True)
                                tmp_level.index = tmp_level.index + 1
                                current_level[['ask_p', 'ask_v']] = tmp_level.copy()
                        else:
                            current_level = make_deal(price, volume, current_level, 2)
                            initiate_side = 2
                    else:
                        if price in current_level['ask_p'].values:
                            current_level.loc[current_level['ask_p'] == price, 'ask_v'] = current_level.loc[
                                                                                              current_level[
                                                                                                  'ask_p'] == price, 'ask_v'] + volume
                        else:
                            current_level.loc[len(current_level.index) + 1] = [price, np.nan, volume, np.nan]
                            tmp_level = current_level[['ask_p', 'ask_v']].copy()
                            tmp_level = tmp_level.sort_values(by='ask_p', ascending=True)
                            tmp_level = tmp_level.reset_index(drop=True)
                            tmp_level.index = tmp_level.index + 1
                            current_level[['ask_p', 'ask_v']] = tmp_level.copy()
                else:
                    warnings.warn('order表出现非1和2的数值')
                    print(file_name, price, volume, order_num, 'order表出现非1和2的数值')

                level_tail = gen_tail(current_level)

            elif sheet == 1:
                if direction == '4':
                    # 撤单
                    if t_bid_num == 0:  # 卖方撤单
                        current_level.loc[current_level['ask_p'] == price, 'ask_v'] = current_level.loc[current_level[
                                                                                                            'ask_p'] == price, 'ask_v'] - volume
                        if len(current_level.loc[current_level['ask_p'] == price]) == 0:
                            # 找不到对应
                            warnings.warn(
                                '撤单找不到对应.【日期】' + str(file_name) + ',【编号】' + str(t_ask_num) + ',【时间】' + str(time))

                        elif current_level.loc[current_level['ask_p'] == price, 'ask_v'].iloc[0] == 0:
                            # 全撤完了
                            tmp_current_level = current_level[['ask_p', 'ask_v']].drop(
                                current_level[current_level['ask_v'] == 0].index).copy()
                            tmp_current_level = tmp_current_level.reset_index(drop=True)
                            tmp_current_level.index = tmp_current_level.index + 1
                            current_level[['ask_p', 'ask_v']] = tmp_current_level
                    elif t_ask_num == 0:  # 买方撤单
                        current_level.loc[current_level['bid_p'] == price, 'bid_v'] = current_level.loc[current_level[
                                                                                                            'bid_p'] == price, 'bid_v'] - volume
                        if len(current_level.loc[current_level['bid_p'] == price]) == 0:
                            warnings.warn(
                                '撤单找不到对应.【日期】' + str(file_name) + ',【编号】' + str(t_bid_num) + ',【时间】' + str(time))

                        elif current_level.loc[current_level['bid_p'] == price, 'bid_v'].iloc[0] == 0:
                            tmp_current_level = current_level[['bid_p', 'bid_v']].drop(
                                current_level[current_level['bid_v'] == 0].index).copy()
                            tmp_current_level = tmp_current_level.reset_index(drop=True)
                            tmp_current_level.index = tmp_current_level.index + 1
                            current_level[['bid_p', 'bid_v']] = tmp_current_level
                    level_tail = gen_tail(current_level)

                elif direction == 'F':
                    if time == 92500000:
                        # 集合竞价阶段
                        if flag_925:
                            call_auction = trade_spot[trade_spot['OrigTime'] == 92500000]
                            tmp_level = order_spot[order_spot['OrigTime'] <= 92500000]
                            for auc_bid_925, auc_ask_925, auc_v in list(
                                    zip(call_auction['BidApplSeqNum'], call_auction['AskApplSeqNum'],
                                        call_auction['TradeQty'])):
                                auc_price = tmp_level.loc[tmp_level['ApplSeqNum'] == auc_bid_925, 'Price'].iloc[0]
                                current_level.loc[current_level['bid_p'] == auc_price, 'bid_v'] = current_level.loc[
                                                                                                      current_level[
                                                                                                          'bid_p'] == auc_price, 'bid_v'] - auc_v
                                auc_price = tmp_level.loc[tmp_level['ApplSeqNum'] == auc_ask_925, 'Price'].iloc[0]
                                current_level.loc[current_level['ask_p'] == auc_price, 'ask_v'] = current_level.loc[
                                                                                                      current_level[
                                                                                                          'ask_p'] == auc_price, 'ask_v'] - auc_v
                                count_0 = len(current_level[current_level['ask_v'] == 0])
                                current_level[['ask_p', 'ask_v']] = current_level[['ask_p', 'ask_v']].shift(
                                    -count_0).copy()
                                count_0 = len(current_level[current_level['bid_v'] == 0])
                                current_level[['bid_p', 'bid_v']] = current_level[['bid_p', 'bid_v']].shift(
                                    -count_0).copy()
                            flag_925 = False
                        level_tail_925 = gen_tail(current_level)
                        output_df.loc[len(output_df.index)] = [time, price, volume, t_bid_num, t_ask_num,
                                                               0] + level_tail_925
                        level_tail = level_tail_925

                    else:
                        # 碰到成交，就把当前订单簿状态写入(这里有个问题：万一还没等上一笔订单交易完成，下一笔就进来了怎么办？)
                        level_tail = gen_tail(current_level)
                        output_df.loc[len(output_df.index)] = [time, price, volume, t_bid_num, t_ask_num,
                                                               initiate_side] + level_tail
                else:
                    warnings.warn('trade表出现非F和4的数值')
                    print(file_name, time, price, volume, direction, 'trade表出现非F和4的数值')
            else:
                warnings.warn('sheet名出现非0 or 1')

            current_level = current_level.dropna(how='all')
            # print(current_level.loc[1:10])
        print(file_name + '匹配完成')

        # =======================================指标构造===========================================

        # 时间窗口长度(分钟)
        time_window = 5  # 5min

        # 波动率取样长度(分钟)
        rv_period = 30  # 30min

        tmp_df = output_df.copy()
        tmp_index = 0

        feature_col = ['volume_dva', 'volume_period_sum', 'rv', 'ImediPrImpac', 'LongtPrImpac', 'duration']
        tmp_df[feature_col] = np.nan

        # 瞬时股价变化率,最后一笔订单为nan,记得消去
        tmp_df.loc[tmp_df['initiate_side'] == 0, 'ImediPrImpac'] = 0
        tmp_df.loc[tmp_df['initiate_side'] == 1, 'ImediPrImpac'] = np.log10(
            output_df.loc[output_df['initiate_side'] == 1, 'AskP1'].shift(-1)) - np.log10(
            output_df.loc[output_df['initiate_side'] == 1, 'AskP1'])
        tmp_df.loc[tmp_df['initiate_side'] == 2, 'ImediPrImpac'] = np.log10(
            output_df.loc[output_df['initiate_side'] == 2, 'BidP1'].shift(-1)) - np.log10(
            output_df.loc[output_df['initiate_side'] == 2, 'BidP1'])

        # 长期价格影响
        output_df['spread'] = ((output_df['AskP1'] - output_df['BidP1'])/2).copy()
        tmp_df.loc[tmp_df['initiate_side'] == 0, 'LongtPrImpac'] = 0
        tmp_df.loc[tmp_df['initiate_side'] == 1, 'LongtPrImpac'] = output_df.loc[output_df['initiate_side'] == 1, 'spread'].shift(-1) - output_df.loc[output_df['initiate_side'] == 1, 'spread']
        tmp_df.loc[tmp_df['initiate_side'] == 2, 'LongtPrImpac'] = output_df.loc[output_df['initiate_side'] == 2, 'spread'].shift(-1) - output_df.loc[output_df['initiate_side'] == 2, 'spread']

        # 挂单时间，计算成交双方挂单时间更长的一方
        tmp_trade_1 = pd.merge(output_df['t_bid_num'], order_spot[['ApplSeqNum', 'OrigTime']], left_on='t_bid_num',
                               right_on=['ApplSeqNum'], how='left')
        tmp_trade_2 = pd.merge(output_df['t_ask_num'], order_spot[['ApplSeqNum', 'OrigTime']], left_on='t_ask_num',
                               right_on=['ApplSeqNum'], how='left')
        tmp_trade_1['time_1'] = tmp_df['time'] - tmp_trade_1['OrigTime']
        tmp_trade_1['time_2'] = tmp_df['time'] - tmp_trade_2['OrigTime']
        tmp_df['duration'] = tmp_trade_1[['time_1', 'time_2']].max(axis=1)
        del tmp_trade_1, tmp_trade_2

        for time, price, volume, ask1, bid1, initiate_side in list(
                zip(tmp_df['time'], tmp_df['price'], tmp_df['volume'], tmp_df['AskP1'], tmp_df['BidP1'],
                    tmp_df['initiate_side'])):
            # 获取time_window前的时间
            current_time = datetime.datetime.strptime(str(int(time)), "%H%M%S%f")
            gap_time = current_time - datetime.timedelta(minutes=time_window)
            num_gap_time = int(gap_time.strftime("%H%M%S%f")) / 1000  # 除以1000是因为输入的微秒有3个0；datetime的微秒有6个0

            # 成交量/时间窗口内平均成交量
            tmp_df.loc[tmp_index, 'volume_dva'] = volume / output_df.loc[
                (output_df['time'] > num_gap_time) & (output_df['time'] <= time), 'volume'].mean()

            # 时间窗口内总成交量
            tmp_df.loc[tmp_index, 'volume_period_sum'] = output_df.loc[
                (output_df['time'] > num_gap_time) & (output_df['time'] <= time), 'volume'].sum()

            # 波动率,to be continued
            # 提取时间窗口内股票tick数据
            gap_time = current_time - datetime.timedelta(minutes=rv_period)
            num_gap_time = int(gap_time.strftime("%H%M%S%f")) / 1000  # 除以1000是因为输入的微秒有3个0；datetime的微秒有6个0

            windowed_snap = snap_level_spot.loc[
                (snap_level_spot['OrigTime'] > num_gap_time) & (snap_level_spot['OrigTime'] <= time), ['OrigTime',
                                                                                                       'AskP1',
                                                                                                       'BidP1']]

            rv_list = list()

            # RV
            rv_list.append(np.sum(np.square(get_rv(windowed_snap, 3))))
            rv_list.append(np.sum(np.square(get_rv(windowed_snap, 9))))
            rv_list.append(np.sum(np.square(get_rv(windowed_snap, 30))))
            rv_list.append(np.sum(np.square(get_rv(windowed_snap, 60))))

            # RV^bc
            r_df = (np.square(get_rv(windowed_snap, 3)))
            eps_squared = np.mean(np.square(r_df))
            bias = len(r_df) / (len(r_df) - 1) * eps_squared
            rv_list.append(np.sum(r_df - bias))

            tmp_df.loc[tmp_index, 'rv'] = sum(rv_list) / len(rv_list)

            tmp_index = tmp_index + 1

        tmp_df['rv'].replace(np.inf, 0, inplace=True)
        output_df_2 = tmp_df[['time', 'price', 'volume', 'initiate_side', 'AskP1', 'BidP1'] + feature_col]
        # 指标归一化,价格冲击率除外
        process_feature_col = ['volume_dva', 'volume_period_sum', 'rv', 'duration']
        output_df_2[['volume'] + process_feature_col] = (output_df_2[['volume'] + process_feature_col].copy() -
                                                         output_df_2[
                                                             ['volume'] + process_feature_col].mean()) / output_df_2[
                                                            ['volume'] + process_feature_col].std()
        print(file_name + '特征计算完成')

        output_df_2.to_csv(root_path + '/dfs/' + stock_index + '/' + stock_index + '+' + file_name + '+feature.csv',
                           index=False, sep=',')


for this_stock_index in stock_index:
    main(this_stock_index)