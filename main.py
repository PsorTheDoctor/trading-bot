from collections.abc import Callable
import time
import argparse

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

ALGORITHMS: dict[str, Callable[[BaseTrader], None]] = {
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

CLI_STRATEGY_PARAM_NAME = 'strategy'
CLI_TRADER_PARAM_NAME = 'trader'

def read_cli_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(f"--{CLI_STRATEGY_PARAM_NAME}", default=DEFAULT_ALGORITHM_NAME, help='Which trading strategy should be used (MACD renko is default)')
    parser.add_argument(f"--{CLI_TRADER_PARAM_NAME}", default=DEFAULT_TRADER_NAME, help='Which FOREX trader should be used (MetaTrader 5 is default)')
    
    args = parser.parse_args()
    
    return vars(args)

def get_trading_strategy(strategy_name: str):
    strategy = ALGORITHMS.get(strategy_name)
    
    if not strategy:
        raise Exception(f"Unsupported algorithm: {strategy_name}")
    
    return strategy

def get_trader(trader_name: str):
    trader = TRADERS.get(trader_name)
    
    if not trader:
        raise Exception(f"Unsupported trader platform: {trader_name}")
    
    return trader


if __name__ == '__main__':
    input_arguments = read_cli_arguments()
    
    algorithm = get_trading_strategy(input_arguments[CLI_STRATEGY_PARAM_NAME])
    trader = get_trader(input_arguments[CLI_TRADER_PARAM_NAME])
    
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
