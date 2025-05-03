import MetaTrader5 as mt5
from pathlib import Path
from os import path

from utils.traders.base_trader import BaseTrader

class BossaTrader(BaseTrader):
    def connect_with_trader(self):
        key = open(path.join(Path(__file__ ).parent.parent.parent,'bossa_trader_key.txt'), 'r').read().split()
        terminal_path = r'C:\Program Files\BOSSAFX 5\terminal64.exe'

        if mt5.initialize(path=terminal_path, login=int(key[0]), password=key[1], server=key[2]):
            print('Connected')
    
    def market_order(self, symbol, vol, buy_sell):
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


    def limit_order(self, symbol, vol, buy_sell, pips_away):
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
