import traceback
import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
from utils.data_loaders import *
from utils.orders import market_order
from utils.technical_indicators import *

pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD']
pos_size = 0.5  # max capital allocated for any currency pair


def trade_signal(merged_df, long_short):
    signal = ''
    df = copy.deepcopy(merged_df)

    if long_short == '':
        if df['bar_num'].tolist()[-1] >= 2 and df['macd'].tolist()[-1] > df['macd_sig'].tolist()[-1]:
            signal = 'buy'
        elif df['bar_num'].tolist()[-1] <= -2 and df['macd'].tolist()[-1] < df['macd_sig'].tolist()[-1]:
            signal = 'sell'
    elif long_short == 'long':
        if df['bar_num'].tolist()[-1] <= -2 and df['macd'].tolist()[-1] < df['macd_sig'].tolist()[-1]:
            signal = 'close_sell'
        elif df['macd'].tolist()[-1] < df['macd_sig'].tolist()[-1] and df['macd'].tolist()[-2] > df['macd_sig'].tolist()[-2]:
            signal = 'close'
    elif long_short == 'short':
        if df['bar_num'].tolist()[-1] >= 2 and df['macd'].tolist()[-1] > df['macd_sig'].tolist()[-1]:
            signal = 'close_buy'
        elif df['macd'].tolist()[-1] > df['macd_sig'].tolist()[-1] and df['macd'].tolist()[-2] < df['macd_sig'].tolist()[-2]:
            signal = 'close'

    return signal


def macd_renko():
    try:
        open_pos = get_positions()
        for currency in pairs:
            long_short = ''
            if len(open_pos) > 0:
                open_pos_cur = open_pos[open_pos['symbol'] == currency]
                if len(open_pos_cur) > 0:
                    if (open_pos_cur.type * open_pos_cur.volume).sum() > 0:
                        long_short = 'long'
                    elif (open_pos_cur.type * open_pos_cur.volume).sum() < 0:
                        long_short = 'short'

            ohlc = get_5m_candles(currency)
            signal = trade_signal(renko_merge(ohlc), long_short)

            if signal == 'buy' or signal == 'sell':
                market_order(currency, pos_size, signal)
                print(f'New {signal} initiated for {currency}')
            elif signal == 'close':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                if total_pos > 0:
                    market_order(currency, total_pos, 'sell')
                elif total_pos < 0:
                    market_order(currency, abs(total_pos), 'buy')
                print(f'All positions closed for {currency}')
            elif signal == 'close_buy':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                market_order(currency, abs(total_pos) + pos_size, 'buy')
                print(f'Existing short position closed for {currency}')
                print(f'New long position initiated for {currency}')
            elif signal == 'close_sell':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                market_order(currency, total_pos + pos_size, 'sell')
                print(f'Existing long position closed for {currency}')
                print(f'New short position initiated for {currency}')

    except Exception as e:
        print("Exception")
        print(traceback.format_exc())
        print(e)
