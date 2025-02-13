import MetaTrader5 as mt5


def market_order(symbol, vol, buy_sell):
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


def limit_order(symbol, vol, buy_sell, pips_away):
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
