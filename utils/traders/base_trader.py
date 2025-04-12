from abc import ABC, abstractmethod


class BaseTrader(ABC):    
    @abstractmethod
    def connect_with_trader(self):
        pass
    
    @abstractmethod
    def market_order(self, symbol, vol, buy_sell):
        pass
    
    @abstractmethod
    def limit_order(self,symbol, vol, buy_sell, pips_away):
        pass