import traceback
import numpy as np
import pandas as pd

from strategies.qlearning_basic import BaseQLearningTrader
from utils.data_loaders import get_positions,get_5m_candles
from utils.constants import CURRENCY_COLUMN_NAME, CURRENCY_PAIRS, TradeAction

MAX_TRADES_PER_ALGORITHM_ITERATION = 5

class QLearningTrader(BaseQLearningTrader):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        super().__init__(alpha, gamma, epsilon, num_states)

    def fill_q_table(self, prices):
        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(prices)
        
        print(f"min price: {min_price}")
        
        for i in range(1, len(prices)):
            current_price = prices[i-1]
            next_price = prices[i]

            # Choose action using epsilon-greedy strategy
            action = self.choose_action(current_price, min_price, max_price)

            # Calculate the reward
            reward = self.calculate_reward(current_price, action, next_price)

            # Update Q-table
            self.update_q_table(current_price, action, reward, next_price, min_price, max_price)

    def update_q_table(self, state, action: TradeAction, reward, next_state, min_price, max_price):
        state_idx = self.get_state_index(state, min_price, max_price)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state, min_price, max_price)

        # Q-value update formula
        best_future_q = np.max(self.q_table[next_state_idx])  # Best Q-value for next state
        self.q_table[state_idx, action_idx] = (1 - self.alpha) * self.q_table[state_idx, action_idx] + \
                                                self.alpha * (reward + self.gamma * best_future_q)
                                                
        print(f"q-table={self.q_table}")

def get_positions_for_currency(all_positions: pd.DataFrame, currency: str) -> pd.DataFrame:
    return all_positions.loc[all_positions[CURRENCY_COLUMN_NAME] == currency]

def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame) -> None:
    # We want to create new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = QLearningTrader()
            
    agent.train(data_for_currency)
        
    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')
            
    agent.perform_trading(data_for_currency, currency)

def qlearning():    
    try:                    
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")
            
        for currency in CURRENCY_PAIRS:
            print(f"Running Q-learning for currency={currency}")
            
            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency)
            
            
    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())
