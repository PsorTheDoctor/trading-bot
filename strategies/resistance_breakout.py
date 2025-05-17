import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
import traceback

from utils.data_loaders import *
from utils.technical_indicators import atr
from utils.constants import CURRENCY_PAIRS
from utils.traders.base_trader import BaseTrader


def merge_atr_with_roll(df):
    df = copy.deepcopy(df)
    df['atr'] = atr(df, 20)
    df['roll_max_cp'] = df['high'].rolling(20).max()
    df['roll_min_cp'] = df['low'].rolling(20).min()
    df['roll_max_vol'] = df['volume'].rolling(20).max()
    df.dropna(inplace=True)
    return df


def trade_signal(df, long_short):
    signal = ''

    if long_short == '':
        if df['high'].tolist()[-1] >= df['roll_max_cp'].tolist()[-1] and df['volume'].tolist()[-1] > 1.5 * df['roll_max_vol'].tolist()[-2]:
            signal = 'buy'
        elif df['low'].tolist()[-1] <= df['roll_min_cp'].tolist()[-1] and df['volume'].tolist()[-1] > 1.5 * df['roll_max_vol'].tolist()[-2]:
            signal = 'sell'
    elif long_short == 'long':
        if df['low'].tolist()[-1] < df['close'].tolist()[-2] - df['atr'].tolist()[-2]:
            signal = 'close'
        elif df['low'].tolist()[-1] <= df['roll_min_cp'].tolist()[-1] and df['volume'].tolist()[-1] > 1.5 * df['roll_max_vol'].tolist()[-2]:
            signal = 'close_sell'
    elif long_short == 'short':
        if df['high'].tolist()[-1] > df['close'].tolist()[-2] + df['atr'].tolist()[-2]:
            signal = 'close'
        elif df['high'].tolist()[-1] >= df['roll_max_cp'].tolist()[-1] and df['volume'].tolist()[-1] > 1.5 * df['roll_max_vol'].tolist()[-2]:
            signal = 'close_buy'

    return signal


def resistance_breakout(trader: BaseTrader):
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
            signal = trade_signal(merge_atr_with_roll(ohlc), long_short)

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
