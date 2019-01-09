'''
@author: Yiyang Wen
@date: 2019/01/04
@python version: 3.6.6
'''
import pandas as pd
import time as t
import math
import numpy as np
from numba import jit
import pyarrow.parquet as pq
import os
import datetime as datetime

class order(object):
    def __init__(self, direction, open_time, contract_value, contract1_open_prc, contract2_open_prc, 
                 stop_profit=100, stop_loss=-100, stop_time_int=4*60*120):
        self.direction = direction
        self.contract_value = contract_value
        self.open_time = open_time
        self.contract1_open_prc = contract1_open_prc
        self.contract2_open_prc = contract2_open_prc
        self.stop_profit = stop_profit
        self.stop_loss = stop_loss
        self.stop_time = open_time + stop_time_int
        self.cost_rate = 0.0
        self.margin_rate = 0.1
        self._open()

    def _open(self):
        # print('open')
        self.contract1_open_pos = self.contract_value / 2.0 / self.contract1_open_prc
        self.contract2_open_pos = self.contract_value / 2.0 / self.contract2_open_prc
        # self.contract1_open_cost = round(self.cost_rate * self.contract1_open_pos, 8)
        # self.contract2_open_cost = round(self.cost_rate * self.contract2_open_pos, 8)
        self.margin = round(self.margin_rate * self.contract_value, 8)

    def is_close(self, contract1_prc, contract2_prc, time):
        if self.direction == 'long':
            ret = (contract1_prc / self.contract1_open_prc + self.contract2_open_prc / contract2_prc) / 2 - 1
            if ret > self.stop_profit or ret < self.stop_loss or time > self.stop_time:
#                 print("强制平仓")
                return True

        elif self.direction == 'short':
            ret = (contract2_prc / self.contract2_open_prc + self.contract1_open_prc / contract1_prc) / 2 - 1
            if ret > self.stop_profit or ret < self.stop_loss or time > self.stop_time:
#                 print("强制平仓")
                return True

        return False

    def close(self, contract1_prc, contract2_prc, time):
        self.contract1_close_prc = contract1_prc
        self.contract2_close_prc = contract2_prc
        self.close_time = time
        self.contract1_close_pos = self.contract1_open_pos
        self.contract2_close_pos = self.contract2_open_pos
        # self.contract1_close_cost = round(self.cost_rate * self.contract1_close_pos, 8)
        # self.contract2_close_cost = round(self.cost_rate * self.contract2_close_pos, 8)
        if self.direction == "long":
            profit = self.contract1_close_pos * self.contract1_close_prc - self.contract2_close_pos * self.contract2_close_prc 
        else:
            profit = -self.contract1_close_pos * self.contract1_close_prc + self.contract2_close_pos * self.contract2_close_prc 
        self.profit = round(profit, 8)
        # self.gain = round(
            # self.profit - self.contract1_open_cost - self.contract2_open_cost - self.contract1_close_cost - self.contract2_close_cost, 8)
        self.gain = self.profit   
        self.profit_rate = self.profit / self.margin
#         print(self.direction, self.contract_value, self.gain, self.open_time, self.contract1_open_prc, self.contract2_open_prc,
#               self.close_time, self.contract1_close_prc, self.contract2_close_prc)

    def value(self, contract1_prc, contract2_prc):
        if self.direction == 'long':
            return round(self.margin + self.contract1_open_pos * contract1_prc 
                         - self.contract2_open_pos * contract2_prc , 8)
        else:
            return round(self.margin - self.contract1_open_pos * contract1_prc 
                         + self.contract2_open_pos * contract2_prc,  8)

class BackTest(object):
    def __init__(self, base_vol, open_threshold, open_threshold_std, close_threshold, close_threshold_std,
                 stop_profit=10000, stop_loss=-10000, stop_time_int=60*4):
        # 初始资金
        self._init = 10000
        # 总利润
        self._profit = 0
        # 当前资金
        self._cur = self._init
        # 当前资产价值
        self._value = self._cur
        # 最大的每次下单的合约数目
        self._base_vol = base_vol
        # 持有订单记录
        self._order_list = []
        self._closed_order_list = []
        # 开仓条件，做空contract1做多contract2的价差门限和std门限
        self._open_threshold = open_threshold
        self._open_threshold_std = open_threshold_std
        # 平仓条件
        self._close_threshold = close_threshold
        self._close_threshold_std = close_threshold_std
        # 交易类型标识 0无操作 1表示交易
        self._trade = 0
        # 设置交易间隔时长 一旦交易 间隔2s
        self._interval = 0
        # 设置资金连一张合约都无法购买的flag False表示资金不足
        self._flag = True
        # 计数开仓次数和平仓次数
        self._counter_open = 0
        self._counter_close = 0
        # 保证金比例
        self._margin_rate = 0.1
        # 手续费
        self._cost_rate = 0.000
        #止盈止损时间终止
        self.stop_profit = stop_profit
        self.stop_loss = stop_loss
        self.stop_time_int = stop_time_int

    @jit(cache=True)
    def isEnough(self, contract_value):
        if self._cur < contract_value:
            return False
        else:
            return True

    @jit(cache=True)
    def CalAveOpen(self, contract1, contract2):
        contract1_bid_prc = contract1[0]
        contract1_ask_prc = contract1[1]
        contract2_bid_prc = contract2[0]
        contract2_ask_prc = contract2[1]
        return contract1_ask_prc, contract1_bid_prc, contract2_ask_prc, contract2_bid_prc

    @jit(cache=True)
    def CalAveClose(self, direction, contract1, contract2):
        if direction == "long":
            contract1_bid_prc = contract1[0]
            contract2_ask_prc = contract2[1]
            return contract1_bid_prc, contract2_ask_prc
        elif direction == "short":
            contract1_ask_prc = contract1[1]
            contract2_bid_prc = contract2[0]
            return contract1_ask_prc, contract2_bid_prc
        return 0, 0

    @jit(cache=True)
    def strategy(self, time, contract1, contract2, MEAN, STD):
        # 优先平仓
        order_list = self._order_list.copy()
        for order_ in order_list:
            contract1_prc, contract2_prc = self.CalAveClose(order_.direction, contract1, contract2)
            if contract1_prc > 0 and contract2_prc > 0:
                diff = contract1_prc / contract2_prc - MEAN
                flag = False
                if order_.direction == "long" and diff > self._close_threshold[0] and diff > STD *                         self._close_threshold_std[0]:
                    flag = True
                elif order_.direction == "short" and diff < self._close_threshold[1] and diff < STD *                         self._close_threshold_std[1]:
                    flag = True
                elif order_.is_close(contract1_prc, contract2_prc, time):
                    flag = True
                if flag:
                    order_.close(contract1_prc, contract2_prc, time)
                    self._cur += order_.gain + order_.margin
                    self._trade = 1
                    self._counter_close += 1
                    self._flag = True
                    order_list.remove(order_)
                    self._order_list = order_list
                    self._closed_order_list.append(order_)
                    self._interval = 0
                    return
        # 判断能否开仓
        if self._flag:
            contract_value = min(self._cur * 10, self._base_vol)
            contract1_ask_prc, contract1_bid_prc, contract2_ask_prc, contract2_bid_prc = self.CalAveOpen(contract1, contract2)
            while contract_value > self._base_vol * 0.1:
                if contract1_ask_prc > 0 and contract2_bid_prc > 0:
                    # 尝试买入开多contract1 卖出开空contract2
                    diff = contract1_ask_prc / contract2_bid_prc - MEAN
                    if diff < self._open_threshold[0] and diff < STD * self._open_threshold_std[0]:
                        if self.isEnough(contract_value):
                            order_ = order("long", time, contract_value, contract1_ask_prc, contract2_bid_prc,
                                          self.stop_profit, self.stop_loss, self.stop_time_int)
                            self._trade = 1
                            self._counter_open += 1
                            self._cur -= order_.margin
                            self._order_list.append(order_)
                            self._interval = 0
                            return
                        else:
                            self._flag = False
                if contract1_bid_prc > 0 and contract2_ask_prc > 0:
                    # 尝试卖出开空contract1 买入开多contract2
                    diff = contract1_bid_prc / contract2_ask_prc - MEAN
                    if diff > self._open_threshold[1] and diff > STD * self._open_threshold_std[1]:
                        if self.isEnough(contract_value):
                            order_ = order("short", time, contract_value, contract1_bid_prc, contract2_ask_prc,
                                           self.stop_profit, self.stop_loss, self.stop_time_int)
                            self._trade = 1
                            self._counter_open += 1
                            self._cur -= order_.margin
                            self._order_list.append(order_)
                            self._interval = 0
                            return
                        else:
                            self._flag = False
                contract_value *= 0.75

    def get_value(self, contract1, contract2):
        contract1_prc = contract1[-1]
        contract2_prc = contract2[-1]
        value = 0
        for order_ in self._order_list:
            value += order_.value(contract1_prc, contract2_prc)
        return value + self._cur

    def get_profit(self):
        # 当前资产价值 - 初始资金
        return self._value - self._init

    def get_profit_rate(self):
        # 利润率
        return self._profit / self._init

    def record_log(self, time):
        profit = '%1.8f' % self._profit
        value = '%1.8f' % self._value
        cur = '%1.8f' % self._cur
        counter_open = '%d' % self._counter_open
        counter_close = '%d' % self._counter_close
#         log = [time.strftime(format='%Y-%m-%d-%H:%M:%S.%f')[:-5], profit, value, cur, counter_open, counter_close]
        log = ['%1.5f' % time, profit, value, cur, counter_open, counter_close]
        return log

    def show_info(self):
        print('initial capital:', self._init)
        print('current capital:', self._cur)
        print('profit:', self._profit)
        print('profit rate:', self.get_profit_rate())

    def trade(self, time, contract1, contract2, MEAN, STD):
        self._trade = 0
        if self._interval == 0:
            self.strategy(time, contract1, contract2, MEAN, STD)
        else:
            self._interval -= 1
        self._value = self.get_value(contract1, contract2)
        self._profit = self.get_profit()
        log = self.record_log(time)
#         if self._trade != 0:
#             print(log)
        return log

