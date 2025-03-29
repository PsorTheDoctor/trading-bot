from enum import Enum

class TradeAction(Enum):
    BUY = 'buy'
    SELL = 'sell'
    HOLD = 'hold'
    CLOSE_POSITIONS = 'close_positions'

CURRENCY_COLUMN_NAME = 'symbol'
CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD']
POSITION_SIZE = 0.5  # max capital allocated for any currency pair
