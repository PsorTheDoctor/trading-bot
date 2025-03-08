import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import datetime as dt


"""
Returns current account balance value (usually the currency is USD)
"""
def get_current_balance() -> float:
    acc_info = mt5.account_info()
    return acc_info.balance

def get_positions_historical(symbol: str, date: dt.datetime, candles: int) -> pd.DataFrame:
    hist_data = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M15, date, candles)
    print(f"hist data: {hist_data}")
    hist_data_df = pd.DataFrame(hist_data)
    print(f"hist data df: {hist_data_df}")
    hist_data_df.time = pd.to_datetime(hist_data_df.time, unit="s")
    hist_data_df.set_index("time", inplace=True)
    
    return hist_data_df

def get_positions() -> pd.DataFrame:
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
