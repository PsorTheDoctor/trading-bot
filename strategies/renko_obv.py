import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
import traceback

from utils.data_loaders import *
from utils.technical_indicators import *
from utils.constants import CURRENCY_PAIRS
from utils.traders.base_trader import BaseTrader


def merge_renko_with_obv():
    df = copy.deepcopy(df)
    df['date'] = df.index
    renko_df = renko(df)
    renko_df.columns = ['date', 'open', 'high', 'low', 'close', 'uptrend', 'bar_num']

    renko_df_to_merge = renko_df.loc[:, ['date', 'bar_num']]
    df.date.astype('datetime64[ns]', copy=False)
    renko_df_to_merge.date = renko_df_to_merge.date.astype('datetime64[ns]')

    merged_df = df.merge(renko_df_to_merge, how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['obv'] = obv(merged_df)
    merged_df['obv_slope'] = slope(merged_df, 5)
    return merged_df


def trade_signal(merged_df, long_short):
    signal = ''
    df = copy.deepcopy(merged_df)

    if long_short == '':
        if df['bar_num'].tolist()[-1] >= 2 and df['obv_slope'].tolist()[-1] > 30:
            signal = 'buy'
        elif df['bar_num'].tolist()[-1] <= -2 and df['obv_slope'].tolist()[-1] < -30:
            signal = 'sell'
    elif long_short == 'long':
        if df['bar_num'].tolist()[-1] <= -2 and df['obv_slope'].tolist()[-1] < -30:
            signal = 'close_sell'
        elif df['bar_num'].tolist()[-1] < 2:
            signal = 'close'
    elif long_short == 'short':
        if df['bar_num'].tolist()[-1] >= 2 and df['obv_slope'].tolist()[-1] > 30:
            signal = 'close_buy'
        elif df['bar_num'].tolist()[-1] > -2:
            signal = 'close'

    return signal


def renko_obv(trader: BaseTrader):
    try:
        open_pos = get_positions()
        for currency in CURRENCY_PAIRS:
            long_short = ''
            if len(open_pos) > 0:
                open_pos_cur = open_pos[open_pos['symbol'] == currency]
                if len(open_pos_cur) > 0:
                    if (open_pos_cur.type * open_pos_cur.volume).sum() > 0:
                        long_short = 'long'
                    elif (open_pos_cur.type * open_pos_cur.volume).sum() < 0:
                        long_short = 'short'

            ohlc = get_5m_candles(currency)
            signal = trade_signal(merge_renko_with_macd(ohlc), long_short)

            if signal == 'buy' or signal == 'sell':
                trader.market_order(currency, POSITION_SIZE, signal)
                print(f'New {signal} initiated for {currency}')
            elif signal == 'close':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                if total_pos > 0:
                    trader.market_order(currency, total_pos, 'sell')
                elif total_pos < 0:
                    trader.market_order(currency, abs(total_pos), 'buy')
                print(f'All positions closed for {currency}')
            elif signal == 'close_buy':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                trader.market_order(currency, abs(total_pos) + POSITION_SIZE, 'buy')
                print(f'Existing short position closed for {currency}')
                print(f'New long position initiated for {currency}')
            elif signal == 'close_sell':
                total_pos = (open_pos_cur.type * open_pos_cur.volume).sum()
                trader.market_order(currency, total_pos + POSITION_SIZE, 'sell')
                print(f'Existing long position closed for {currency}')
                print(f'New short position initiated for {currency}')

    except Exception as e:
        print(traceback.format_exc())
        print(e)
