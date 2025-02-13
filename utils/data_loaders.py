import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import datetime as dt


def get_positions():
    positions = mt5.positions_get()

    if len(positions) > 0:
        df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        df.time = pd.to_datetime(df.time, unit='s')
        df.drop(['time_msc', 'time_update', 'time_update_msc', 'external_id'], axis=1, inplace=True)
        # To distinguish between long and short positions
        df.type = np.where(df.type == 0, 1, -1)
    else:
        df = pd.DataFrame()

    return df


def get_orders():
    orders = mt5.orders_get()

    if len(orders) > 0:
        df = pd.DataFrame(orders, columns=orders[0]._asdict().keys())
        df.time = pd.to_datetime(df.time, unit='s')
    else:
        df = pd.DataFrame()

    return df


def get_5m_candles(currency, lookback=10, bars=250):
    data = mt5.copy_rates_from(
        currency, mt5.TIMEFRAME_M5, dt.datetime.now() - dt.timedelta(lookback), bars
    )
    df = pd.DataFrame(data)
    df.time = pd.to_datetime(df.time, unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
    return df
