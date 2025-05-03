import numpy as np
import pandas as pd
import traceback
import random
from typing import Literal

from strategies.temporal_difference_learning import TemporalDifferenceLearning
from utils.data_loaders import get_positions, get_5m_candles
from utils.constants import CURRENCY_PAIRS, CURRENCY_PAIRS, POSITION_SIZE
from utils.traders.base_trader import BaseTrader

ActionType = Literal['buy', 'sell', 'hold']


class SarsaTrader(TemporalDifferenceLearning):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = ['buy', 'sell', 'hold']  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        self.q_table = np.zeros((num_states, len(self.actions)))  # Q-table initialization

    def update_q_table(self, state, action, reward, next_state, next_action, min_price, max_price):
        state_idx = self.get_state_index(state, min_price, max_price)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state, min_price, max_price)
        next_action_idx = self.get_action_index(next_action)

        # Q-table update formula
        next_q_value = self.q_table[next_state_idx, next_action_idx]
        self.q_table[state_idx, action_idx] = (1 - self.alpha) * self.q_table[state_idx, action_idx] + \
          self.alpha * (reward + self.gamma * next_q_value)

    def train(self, data: pd.DataFrame) -> None:
        prices = self.extract_prices_from_data(data)
        print(f"extracted train data: {prices}")

        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(prices)
        print(f"min price: {min_price}")

        # Training on Forex data
        for i in range(1, len(prices)):
            current_price = prices[i - 1]
            next_price = prices[i]

            # Choose action using epsilon-greedy strategy
            action = self.choose_action(current_price, min_price, max_price)

            # Calculate the reward
            reward = self.calculate_reward(current_price, action, next_price)

            # Select next action using the same policy
            next_action = self.choose_action(next_price, min_price, max_price)

            # Update Q-table
            self.update_q_table(current_price, action, reward, next_price, next_action, min_price, max_price)


def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame, trader: BaseTrader) -> None:
    # We want to create a new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = SarsaTrader()
    agent.train(data_for_currency)

    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')

    agent.perform_trading(data_for_currency, currency, trader)


def sarsa(trader):
    try:
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")

        for currency in CURRENCY_PAIRS:
            print(f"Running SARSA for currency={currency}")

            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency, trader)

    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())
