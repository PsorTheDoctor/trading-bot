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
    
    renko_df_to_merge = renko_df.loc[:, ['date', 'bar_num']]
    df.date.astype('datetime64[ns]', copy=False)
    renko_df_to_merge.date = renko_df_to_merge.date.astype('datetime64[ns]')
    
    merged_df = df.merge(renko_df_to_merge, how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['macd'] = macd(merged_df, 12, 26, 9)[0]
    merged_df['macd_sig'] = macd(merged_df, 12, 26, 9)[1]
    return merged_df


def obv(df):
    """
    On Balance Volume
    """
    df = df.copy()
    df['daily_ret'] = df['adj close'].pct_change()
    df['direction'] = np.where(df['daily_ret'] >= 0, 1, -1)
    df['direction'][0] = 0
    df['vol_adj'] = df['volume'] = df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


def ema(df, b):
    df = df.copy()
    df['ema'] = df['close'].ewm(span=b, min_periods=b).mean()
    df.dropna(inplace=True)
    return df['ema']


def ichimoku(df, tenkan_sen=9, kijun_sen=26, senkou_span_b=52):
    df = df.copy()
    df['tenkan_sen'] = (df['high'].rolling(window=tenkan_sen).max() + df['low'].rolling(window=tenkan_sen).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=kijun_sen).max() + df['low'].rolling(window=kijun_sen).min()) / 2
    df['chikou_span'] = df['close'].shift(-kijun_sen)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_sen)
    df['senkou_span_b'] = ((df['high'].rolling(window=senkou_span_b).max() + df['low'].rolling(window=senkou_span_b).min()) / 2).shift(kijun_sen)
    df.dropna(inplace=True)
    return df[['tenkan_sen', 'kijun_sen', 'chikou_span', 'senkou_span_a', 'senkou_span_b']]


def stochastic_oscillator(df, k_period=14, d_period=3):
    df = df.copy()
    df['lowest_low'] = df['low'].rolling(window=k_period).min()
    df['highest_high'] = df['high'].rolling(window=k_period).max()
    df['%K'] = (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    df.drop(columns=['lowest_low', 'highest_high'], inplace=True)
    df.dropna(inplace=True)
    return df[['%K', '%D']]


def cci(df, period=20):
    df = df.copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma_tp'] = df['tp'].rolling(window=period).mean()
    df['mad'] = df['tp'].rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    df['CCI'] = (df['tp'] - df['sma_tp']) / (0.015 * df['mad'])
    df.drop(columns=['tp', 'sma_tp', 'mad'], inplace=True)
    df.dropna(inplace=True)
    return df[['CCI']]


def mfi(df, period=14):
    """
    Money Flow Index - podobny do RSI, ale uwzględnia jeszcze wolumen.
    """
    df = df.copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['mf'] = df['tp'] * df['volume']
    df['mf_positive'] = df['mf'].where(df['tp'].diff() > 0, 0)
    df['mf_negative'] = df['mf'].where(df['tp'].diff() < 0, 0)
    df['sum_mf_positive'] = df['mf_positive'].rolling(window=period).sum()
    df['sum_mf_negative'] = df['mf_negative'].rolling(window=period).sum()
    df['mfr'] = df['sum_mf_positive'] / df['sum_mf_negative']
    df['mfi'] = 100 - (100 / (1 + df['mfr']))
    df.dropna(inplace=True)
    return df['mfi']


def donchian_channels(df, period=20):
    """
    Donchian Channels - kanał to trzy linie generowane poniższymi wzorami;
    gdy cena przekroczy górną linię to może sugerować początek trendu wzrostowego, analogicznie odwrotnie
    """
    df = df.copy()
    df['upper'] = df['high'].rolling(window=period).max()
    df['lower'] = df['low'].rolling(window=period).min()
    df['middle'] = (df['upper'] + df['lower']) / 2
    df.dropna(inplace=True)
    return df[['upper', 'lower', 'middle']]


def pivot_points(df):
    """
    Pivot Points - oblicza się poziomy wsparcia i oporu; cena powyżej punktu pivota to trend wzrostowy, poniżej - spadkowy.
    Do tego mamy punkty wsparcia i oporu pokazujące wsparcie/opór
    """
    df = df.copy()
    df['pivot_points'] = (df['high'] + df['low'] + df['close']) / 3
    df['S1'] = 2 * df['pivot_points'] - df['high']
    df['R1'] = 2 * df['pivot_points'] - df['low']
    df['S2'] = df['pivot_points'] - (df['high'] - df['low'])
    df['R2'] = df['pivot_points'] + (df['high'] - df['low'])
    df['S3'] = df['low'] - 2 * (df['high'] - df['pivot_points'])
    df['R3'] = df['high'] + 2 * (df['pivot_points'] - df['low'])
    df.dropna(inplace=True)
    return df[['pivot_points', 'S1', 'R1', 'S2', 'R2', 'S3', 'R3']]
