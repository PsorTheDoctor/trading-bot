import traceback
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

import pandas as pd

from strategies.qlearning_basic import BaseQLearningTrader
from utils.constants import CURRENCY_PAIRS, POSITION_SIZE, TradeAction
from utils.data_loaders import get_5m_candles, get_positions
from utils.orders import market_order

LOT_SIZE = 0.1
EPISODES = 1000
STEPS_PER_EPISODE = 500
BATCH_SIZE = 32

ACTION_TO_TRADE_ACTION_MAPPINGS = {
    0: TradeAction.HOLD,
    1: TradeAction.BUY,
    2: TradeAction.SELL,
    3: TradeAction.CLOSE_POSITIONS
}

# ---------------------------
# Environment for Forex Trading
# ---------------------------
class DeepQLearningTrader(BaseQLearningTrader):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100, state_size=1, action_size=len(ACTION_TO_TRADE_ACTION_MAPPINGS.keys())):
        super().__init__(alpha, gamma, epsilon, num_states)
        self.state_size = state_size
        self.action_size = action_size
        
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Network for Deep Q-Learning using Keras
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def fill_q_table(self, prices):
        # Create an empty Q-table with the right shape
        q_table = np.zeros((len(prices), self.action_size))
        # For each price (state), use the model to predict Q-values
        for i, price in enumerate(prices):
            # Reshape the price into the expected input shape (batch size 1, 1 feature)
            state = np.array([[price]])
            q_values = self.model.predict(state, verbose=0)
            q_table[i] = q_values
        self.q_table = q_table  # Save the Q-table in the class field
        

def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame) -> None:
    # We want to create new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = DeepQLearningTrader()
            
    agent.train(data_for_currency)
        
    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')
            
    agent.perform_trading(data_for_currency, currency)

def deep_qlearning():    
    try:                    
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")
            
        for currency in CURRENCY_PAIRS:
            print(f"Running Deep Q-learning for currency={currency}")
            
            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency)
            
            
    except Exception as e:
        print(f"Unexpected error while trying to perform deep q-learning algorithm: {e}")
        print(traceback.format_exc())
