#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import talib as tl

def check(code_name, data, date=None, threshold=30):
    """
    检查股票是否符合策略条件：
    1. 30日均线多头排列且加速上涨；
    2. 成交量连续3天放大且大于20日均量；
    3. MACD金叉且位于零轴以上，柱状图持续放大；
    4. RSI处于50-70之间；
    5. KDJ金叉且在50以上；
    6. 价格在布林带中轨和上轨之间。

    参数：
    - code_name: 股票代码和名称
    - data: 包含日期、收盘价、成交量等数据的DataFrame
    - date: 截止日期，默认为None
    - threshold: 计算均线的时间周期，默认为30天

    返回：
    - 符合条件返回True，否则返回False
    """
    if date is None:
        end_date = code_name[0]
    else:
        end_date = date.strftime("%Y-%m-%d")

    # 过滤数据，只保留截止日期之前的数据
    if end_date is not None:
        mask = (data['date'] <= end_date)
        data = data.loc[mask].copy()

    # 数据量不足时直接返回False
    if len(data.index) < threshold:
        return False

    # 计算30日均线
    data.loc[:, 'ma30'] = tl.MA(data['close'].values, timeperiod=30)
    data['ma30'].values[np.isnan(data['ma30'].values)] = 0.0

    # 计算成交量20日均量
    data.loc[:, 'vol_ma20'] = tl.MA(data['volume'].values, timeperiod=20)
    data['vol_ma20'].values[np.isnan(data['vol_ma20'].values)] = 0.0

    # 计算MACD
    data.loc[:, 'macd'], data.loc[:, 'macdsignal'], data.loc[:, 'macdhist'] = tl.MACD(
        data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)

    # 计算RSI
    data.loc[:, 'rsi'] = tl.RSI(data['close'].values, timeperiod=14)

    # 新增：计算KDJ指标
    data.loc[:, 'k'], data.loc[:, 'd'] = tl.STOCH(data['high'].values,
                                                  data['low'].values,
                                                  data['close'].values,
                                                  fastk_period=9,
                                                  slowk_period=3,
                                                  slowk_matype=0,
                                                  slowd_period=3,
                                                  slowd_matype=0)

    # 新增：计算布林带
    data.loc[:, 'upper'], data.loc[:, 'middle'], data.loc[:, 'lower'] = tl.BBANDS(
        data['close'].values,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0)

    # 新增：计算5日均线
    data.loc[:, 'ma5'] = tl.MA(data['close'].values, timeperiod=5)
    data['ma5'].values[np.isnan(data['ma5'].values)] = 0.0

    # 新增：计算短期RSI(5)
    data.loc[:, 'rsi5'] = tl.RSI(data['close'].values, timeperiod=5)

    # 新增：计算威廉指标(W%R)
    data.loc[:, 'willr'] = tl.WILLR(data['high'].values,
                                    data['low'].values,
                                    data['close'].values,
                                    timeperiod=14)

    # 新增：计算ATR
    data.loc[:, 'atr'] = tl.ATR(data['high'].values,
                                data['low'].values,
                                data['close'].values,
                                timeperiod=14)

    # 新增：计算5日收益率
    data['return'] = data['close'].pct_change(periods=5)


    # 只保留最近threshold天的数据
    data = data.tail(n=threshold)

    # 计算均线多头排列的条件
    step1 = round(threshold / 3)
    step2 = round(threshold * 2 / 3)

    # 条件1：30日均线多头排列且加速上涨
    # 条件1：优化均线条件，加入5日均线判断
    ma_condition = (data.iloc[-1]['ma5'] > data.iloc[-1]['ma30']) and \
                   (data.iloc[0]['ma5'] < data.iloc[step1]['ma5'] < \
                    data.iloc[step2]['ma5'] < data.iloc[-1]['ma5'])


    # 条件2：优化成交量条件，要求连续3天成交量增加且显著放大
    volume_trend = (data.iloc[-1]['volume'] > data.iloc[-2]['volume'] > data.iloc[-3]['volume'])
    volume_condition = volume_trend and (data.iloc[-1]['volume'] > 1.5 * data.iloc[-1]['vol_ma20'])

    # 条件3：优化MACD条件，加入柱状图力度判断
    macd_strength = data.iloc[-1]['macdhist'] > data.iloc[-2]['macdhist'] > 0
    macd_condition = (data.iloc[-1]['macd'] > data.iloc[-1]['macdsignal']) and \
                     (data.iloc[-1]['macd'] > 0) and macd_strength

    # 条件4：优化RSI条件，加入短期RSI判断
    rsi_condition = (50 < data.iloc[-1]['rsi'] < 70) and \
                    (50 < data.iloc[-1]['rsi5'] < 70)


    # 新增条件5：KDJ金叉且在50以上
    kdj_condition = (data.iloc[-1]['k'] > data.iloc[-1]['d']) and \
                    (data.iloc[-1]['k'] > 50) and (data.iloc[-1]['d'] > 50)

    # 新增条件6：布林带条件，价格在中轨和上轨之间
    bb_condition = (data.iloc[-1]['close'] > data.iloc[-1]['middle']) and \
                   (data.iloc[-1]['close'] < data.iloc[-1]['upper'])

    # 新增条件7：威廉指标条件
    willr_condition = -80 < data.iloc[-1]['willr'] < -20

    # 新增条件8：5日收益率条件
    return_condition = data.iloc[-1]['return'] > 0

    # 新增条件9：ATR波动率条件
    atr_condition = data.iloc[-1]['atr'] < 0.05 * data.iloc[-1]['close']

    # 综合所有条件
    if ma_condition and volume_condition and macd_condition and \
            rsi_condition and kdj_condition and bb_condition and \
            willr_condition and return_condition and atr_condition:
        return True
    else:
        return False