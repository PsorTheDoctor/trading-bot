import MetaTrader5 as mt5
import time
from strategies.portfolio_rebalance import portfolio_rebalance
from strategies.obv_renko import obv_renko
from strategies.macd_renko import macd_renko


if __name__ == '__main__':
    interval = 60  # 1-minute interval in seconds

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
            macd_renko()
            time.sleep(interval - ((time.time() - start) % interval))
        except KeyboardInterrupt:
            print('\nExiting')
            exit()
