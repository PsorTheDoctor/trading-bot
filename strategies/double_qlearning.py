import numpy as np
import pandas as pd
import traceback
import random
from typing import Literal

from strategies.temporal_difference_learning import TemporalDifferenceLearning
from utils.data_loaders import get_positions, get_5m_candles
from utils.constants import CURRENCY_PAIRS, CURRENCY_PAIRS, POSITION_SIZE

ActionType = Literal['buy', 'sell', 'hold']


class DoubleQLearningTrader(TemporalDifferenceLearning):
    def __init__(self, trader, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        self.trader = trader
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = ['buy', 'sell', 'hold']  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        self.q1_table = np.zeros((num_states, len(self.actions)))  # Q1-table initialization
        self.q2_table = np.zeros((num_states, len(self.actions)))  # Q2-table initialization

    def choose_action(self, state, min_price, max_price) -> ActionType:
        # Epsilon-greedy strategy for action selection
        state_idx = self.get_state_index(state, min_price, max_price)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            return self.actions[np.argmax(self.q_table[state_idx])]  # Exploitation

    def update_q_table(self, state, action: ActionType, reward, next_state, min_price, max_price):
        state_idx = self.get_state_index(state, min_price, max_price)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state, min_price, max_price)

        # Randomly select whether to update Q1-table or Q2-table
        if np.random.rand() < 0.5:
            # Q1-table update formula
            best_action_idx = np.argmax(self.q1_table[next_state_idx])
            target_q = self.q2_table[next_state_idx, best_action_idx]
            self.q1_table[state_idx, action_idx] = (1 - self.alpha) * self.q1_table[state_idx, action_idx] + \
                                                   self.alpha * (reward + self.gamma * target_q)
        else:
            # Q2-table update formula
            best_action_idx = np.argmax(self.q2_table[next_state_idx])
            target_q = self.q1_table[next_state_idx, best_action_idx]
            self.q2_table[state_idx, action_idx] = (1 - self.alpha) * self.q2_table[state_idx, action_idx] + \
                                                   self.alpha * (reward + self.gamma * target_q)

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

            # Update Q-tables
            self.update_q_table(current_price, action, reward, next_price, min_price, max_price)

    def perform_trading(self, data: pd.DataFrame, currency: str):
        prices = self.extract_prices_from_data(data)

        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(prices)

        # Test the trained model with new data
        for i in range(1, len(prices)):
            current_price = prices[i - 1]
            action = self.choose_action(current_price, min_price, max_price)
            print(f"action: {action}")

            if action == 'buy' or action == 'sell':
                order_status = self.trader.market_order(currency, POSITION_SIZE, action)
                print(f"order status: {order_status}")


def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame, trader) -> None:
    # We want to create a new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = DoubleQLearningTrader(trader)
    agent.train(data_for_currency)

    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')

    agent.perform_trading(data_for_currency, currency)


def double_qlearning(trader):
    try:
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")

        for currency in CURRENCY_PAIRS:
            print(f"Running Double Q-learning for currency={currency}")

            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency, trader)

    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())
