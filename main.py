import MetaTrader5 as mt5
import pandas as pd
import datetime as dt
import time
from technical_indicators import *

pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD']
pos_size = 0.5  # max capital allocated for any currency pair
interval = 60  # 1-minute interval in seconds


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


def market_order(symbol, vol, buy_sell):
    if buy_sell.lower()[0] == 'b':
        direction = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        direction = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': vol,
        'price': price,
        'type': direction,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_RETURN
    }
    status = mt5.order_send(request)
    return status


def limit_order(symbol, vol, buy_sell, pips_away):
    pip_unit = 10 * mt5.symbol_info(symbol).point

    if buy_sell.lower()[0] == 'b':
        direction = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask + pips_away * pip_unit
    else:
        direction = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid - pips_away * pip_unit

    request = {
        'action': mt5.TRADE_ACTION_PENDING,
        'symbol': symbol,
        'volume': vol,
        'price': price,
        'type': direction,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_RETURN
    }
    status = mt5.order_send(request)
    return status


def main():
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
        print(e)


if __name__ == '__main__':
    key = open('D:\meta_trader_key.txt', 'r').read().split()
    path = r'C:\Program Files\MetaTrader 5\terminal64.exe'

    if mt5.initialize(path=path, login=int(key[0]), password=key[1], server=key[2]):
        print('Connected')

    start = time.time()
    timeout = time.time() + 3600
    while time.time() <= timeout:
        try:
            print('Passthrough at', time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(time.time())
            ))
            main()
            time.sleep(interval - ((time.time() - start) % interval))
        except KeyboardInterrupt:
            print('\nExiting')
            exit()
