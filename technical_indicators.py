import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy


def macd(df, a, b, c):
    """
    Moving Average Convergence / Divergence
    Typical values are: a=12, b=26, c=9
    """
    df = df.copy()
    df['ma_fast'] = df['close'].ewm(span=a, min_periods=a).mean()
    df['ma_slow'] = df['close'].ewm(span=b, min_periods=b).mean()
    df['macd'] = df['ma_fast'] - df['ma_slow']
    df['signal'] = df['macd'].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return (df['macd'], df['signal'])


def atr(df, n):
    """
    Average True Range
    """
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1, skipna=False)
    df['atr'] = df['tr'].rolling(n).mean()
    df = df.drop(['h-l', 'h-pc', 'l-pc'], axis=1)
    return df


def slope(points, n):
    slopes = [i * 0 for i in range(n - 1)]

    for i in range(n, len(points) + 1):
        y = points[i - n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])

    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)


def renko(df):
    df = df.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    df2 = Renko(df)
    df2.brick_size = round(atr(df, 120)['atr'].iloc[-1], 4)
    renko_df = df2.get_ohlc_data()
    renko_df['bar_num'] = np.where(
        renko_df['uptrend'] == True, 1, np.where(renko_df['uptrend'] == False, -1, 0)
    )
    for i in range(1, len(renko_df['bar_num'])):
        if renko_df.loc[i, 'bar_num'] > 0 and renko_df.loc[i - 1, 'bar_num'] > 0:
            renko_df.loc[i, 'bar_num'] += renko_df.loc[i - 1, 'bar_num']
        elif renko_df.loc[i, 'bar_num'] < 0 and renko_df.loc[i - 1, 'bar_num'] < 0:
            renko_df.loc[i, 'bar_num'] += renko_df.loc[i - 1, 'bar_num']

    renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
    return renko_df


def renko_merge(df):
    df = copy.deepcopy(df)
    df['date'] = df.index
    renko_df = renko(df)
    renko_df.columns = ['date', 'open', 'high', 'low', 'close', 'uptrend', 'bar_num']
    merged_df = df.merge(renko_df.loc[:, ['date', 'bar_num']], how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['macd'] = macd(merged_df, 12, 26, 9)[0]
    merged_df['macd_sig'] = macd(merged_df, 12, 26, 9)[1]
    return merged_df
