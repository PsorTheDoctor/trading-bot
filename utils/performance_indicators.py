import numpy as np


def cagr(df, indicator_parameter):
    """
    Cumulative Annual Growth Rate
    """
    df = df.copy()
    df['cum_return'] = (1 + df['ret']).cumprod()
    n = len(df) / indicator_parameter
    cagr = (df['cum_return'].tolist()[-1]) ** (1 / n) - 1
    return cagr


def volatility(df, indicator_parameter):
    df = df.copy()
    vol = df['ret'].std() * np.sqrt(indicator_parameter)
    return vol


def sharpe(df, rf, indicator_parameter):
    df = df.copy()
    sharpe = (cagr(df, indicator_parameter) - rf) / volatility(df, indicator_parameter)
    return sharpe


def max_drawdown(df):
    df = df.copy()
    df['cum_return'] = (1 + df['ret']).cumprod()
    df['cum_roll_max'] = df['cum_return'].cummax()
    df['drawdown'] = df['cum_roll_max'] - df['cum_return']
    df['drawdown_pct'] = df['drwadown'] / df['cum_roll_max']
    max_drawdown = df['drawdown_pct'].max()
    return max_drawdown
