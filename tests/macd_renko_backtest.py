import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import datetime as dt
from utils.technical_indicators import *
from utils.performance_indicators import *

tickers = ['MSFT', 'AAPL', 'AMZN']
ohlc = {}  # dict with ohlc value for each stock
key = open(r'D:\alpha_vantage_key.txt', 'r').read()
ts = TimeSeries(key=key, output_format='pandas')
INDICATOR_PARAMETER = 252 * 78


def fetch_data(tickers, provider):
    attempt = 0
    drop = []  # list to store tickers where close price was extracted

    while len(tickers) != 0 and attempt < 1:
        tickers = [j for j in tickers if j not in drop]
        for i in range(len(tickers)):
            try:
                if provider == 'yahoo':
                    start = dt.datetime.today() - dt.timedelta(3650)
                    end = dt.datetime.today()
                    ohlc[tickers[i]] = yf.download(tickers[i], start, end, interval='1mo')
                elif provider == 'alpha_vantage':
                    ohlc[tickers[i]] = ts.get_intraday(
                        symbol=tickers[i], interval='5min', outputsize='full', extended_hours=True
                    )[0]
                else:
                    print('Wrong provider')
                ohlc[tickers[i]].columns = ['open', 'high', 'low', 'adj close', 'volume']
                drop.append(tickers[i])
            except:
                print(f'Failed to fetch {tickers[i]} data')
                continue
        attempt += 1
    return ohlc


"""
Backtesting
"""
ohlc = fetch_data(tickers, provider='yahoo')
tickers = ohlc.keys()
ohlc_renko = {}
df = copy.deepcopy(ohlc)
tickers_signal = {}
tickers_ret = {}

for ticker in tickers:
    renko_df = renko(df[ticker])
    renko_df.columns = ['date', 'open', 'high', 'low', 'close', 'uptrend', 'bar_num']
    df[ticker]['date'] = df[ticker].index
    ohlc_renko[ticker] = df[ticker].merge(
        renko.loc[:, ['date', 'bar_num']], how='outer', on='Date'
    )
    ohlc_renko[ticker]['bar_num'].fillna(method='ffill', inplace=True)
    ohlc_renko[ticker]['macd'] = macd(ohlc_renko[ticker], 12, 26, 9)[0]
    ohlc_renko[ticker]['macd_sig'] = macd(ohlc_renko[ticker], 12, 26, 9)[1]
    ohlc_renko[ticker]['macd_slope'] = slope(ohlc_renko[ticker]['macd'], 5)
    ohlc_renko[ticker]['macd_sig_slope'] = slope(ohlc_renko[ticker]['macd_sig'], 5)
    tickers_signal[ticker] = ''
    tickers_ret[ticker] = []

# Identifying signals and calculating daily return
for ticker in tickers:
    print(f'Calculating daily return for {ticker}')
    for i in range(len(ohlc[ticker])):
        if tickers_signal[ticker] == '':
            tickers_ret[ticker].append(0)
            if i > 0:
                if ohlc_renko[ticker]['bar_num'][i] >= 2 and \
                    ohlc_renko[ticker]['macd'][i] > ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] > ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = 'buy'
                elif ohlc_renko[ticker]['bar_num'][i] <= -2 and \
                    ohlc_renko[ticker]['macd'][i] < ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] < ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = 'sell'
        elif tickers_signal[ticker] == 'buy':
            tickers_ret[ticker].append(
                (ohlc_renko[ticker]['adj close'][i] / ohlc_renko[ticker]['adj close'][i - 1]) - 1
            )
            if i > 0:
                if ohlc_renko[ticker]['bar_num'][i] <= -2 and \
                    ohlc_renko[ticker]['macd'][i] < ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] < ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = 'sell'
                elif ohlc_renko[ticker]['macd'][i] < ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] < ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = ''
        elif tickers_signal[ticker] == 'sell':
            tickers_ret[ticker].append(
                (ohlc_renko[ticker]['adj close'][i - 1] / ohlc_renko[ticker]['adj close'][i]) - 1
            )
            if i > 0:
                if ohlc_renko[ticker]['bar_num'][i] >= 2 and \
                    ohlc_renko[ticker]['macd'][i] > ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] > ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = 'buy'
                elif ohlc_renko[ticker]['macd'][i] > ohlc_renko[ticker]['macd_sig'][i] and \
                    ohlc_renko[ticker]['macd_slope'][i] > ohlc_renko[ticker]['macd_sig_slope'][i]:
                    tickers_signal[ticker] = ''

    ohlc_renko[ticker]['ret'] = np.array(tickers_ret[ticker])

# Calculate overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]['ret']

# Visualize strategy returns
strategy_df['ret'] = strategy_df.mean(axis=1)
(1 + strategy_df['ret']).cumprod().plot()

# Calculate individual stock's KPIs
cagr_dict = {}
sharpe_dict = {}
max_drawdown_dict = {}
for ticker in tickers:
    print(f'Calculating KPIs for {ticker}')
    cagr_dict[ticker] = cagr(ohlc_renko[ticker], INDICATOR_PARAMETER)
    sharpe_dict[ticker] = sharpe(ohlc_renko[ticker], 0.025, INDICATOR_PARAMETER)
    max_drawdown_dict[ticker] = max_drawdown(ohlc_renko[ticker])

kpi_df = pd.DataFrame([cagr_dict, sharpe_dict, max_drawdown_dict],
    index=['return', 'sharpe ratio', 'max drawdown']
)
