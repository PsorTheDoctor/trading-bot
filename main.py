import MetaTrader5 as mt5
import time
import sys

from strategies.macd_renko import macd_renko
from strategies.qlearning import qlearning
from strategies.deep_qlearning import deep_qlearning
from strategies.sarsa import sarsa

ALGORITHMS = {
    'renko': macd_renko,
    'qlearning': qlearning,
    'deep-qlearning': deep_qlearning,
    'sarsa': sarsa,
}
DEFAULT_ALGORITHM_NAME = 'renko'


if __name__ == '__main__':
    algorithm_name = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_ALGORITHM_NAME
    algorithm = ALGORITHMS.get(algorithm_name)
    if not algorithm:
        raise Exception(f"Unsupported algorithm: {algorithm_name}")
    
    interval = 60  # 1-minute interval in seconds

    key = open('meta_trader_key.txt', 'r').read().split()
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
            algorithm()
            time.sleep(interval - ((time.time() - start) % interval))
        except KeyboardInterrupt:
            print('\nExiting')
            exit()
