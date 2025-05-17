import MetaTrader5 as mt5
import time
import sys

from strategies.resistance_breakout import resistance_breakout
from strategies.renko_macd import renko_macd
from strategies.renko_obv import renko_obv
from strategies.qlearning import qlearning
from strategies.deep_qlearning import deep_qlearning
from strategies.sarsa import sarsa
from strategies.double_qlearning import double_qlearning
from strategies.expected_sarsa import expected_sarsa

from utils.traders.base_trader import BaseTrader
from utils.traders.bossa_trader import BossaTrader
from utils.traders.metatrader5_trader import MetaTrader5Trader

ALGORITHMS = {
    'resistance_breakout': resistance_breakout,
    'renko_macd': renko_macd,
    'renko_obv': renko_obv,
    'qlearning': qlearning,
    'deep_qlearning': deep_qlearning,
    'sarsa': sarsa,
    'double_qlearning': double_qlearning,
    'expected_sarsa': expected_sarsa
}
DEFAULT_ALGORITHM_NAME = 'renko_macd'

TRADERS: dict[str, BaseTrader] = {
    'mt5': MetaTrader5Trader(),
    'bossa': BossaTrader(),
}
DEFAULT_TRADER_NAME = 'mt5'


if __name__ == '__main__':
    algorithm_name = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_ALGORITHM_NAME
    algorithm = ALGORITHMS.get(algorithm_name)
    if not algorithm:
        raise Exception(f"Unsupported algorithm: {algorithm_name}")
    
    trader_name = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_TRADER_NAME
    trader = TRADERS.get(trader_name)
    if not trader:
        raise Exception(f"Unsupported trader platform: {trader_name}")
    
    interval = 60  # 1-minute interval in seconds

    trader.connect_with_trader()

    start = time.time()
    timeout = time.time() + 3600
    while time.time() <= timeout:
        try:
            print('Passthrough at', time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(time.time())
            ))
            algorithm(trader)
            time.sleep(interval - ((time.time() - start) % interval))
        except KeyboardInterrupt:
            print('\nExiting')
            exit()
