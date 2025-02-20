#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import talib as tl

def check(code_name, data, date=None, threshold=30):
    """
    检查股票是否符合策略条件：
    1. 30日均线多头排列且加速上涨；
    2. 成交量放大；
    3. MACD金叉且位于零轴以上；
    4. RSI处于50-70之间。
    
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

    # 只保留最近threshold天的数据
    data = data.tail(n=threshold)

    # 计算均线多头排列的条件
    step1 = round(threshold / 3)
    step2 = round(threshold * 2 / 3)

    # 条件1：30日均线多头排列且加速上涨
    ma30_condition = (data.iloc[0]['ma30'] < data.iloc[step1]['ma30'] < \
                      data.iloc[step2]['ma30'] < data.iloc[-1]['ma30']) and \
                     (data.iloc[-1]['ma30'] > 1.2 * data.iloc[0]['ma30'])

    # 条件2：成交量放大（当日成交量大于20日均量）
    volume_condition = data.iloc[-1]['volume'] > data.iloc[-1]['vol_ma20']

    # 条件3：MACD金叉且位于零轴以上
    macd_condition = (data.iloc[-1]['macd'] > data.iloc[-1]['macdsignal']) and \
                     (data.iloc[-1]['macd'] > 0)

    # 条件4：RSI处于50-70之间，避免超买
    rsi_condition = 50 < data.iloc[-1]['rsi'] < 70

    # 综合所有条件
    if ma30_condition and volume_condition and macd_condition and rsi_condition:
        return True
    else:
        return False
