import numpy as np
import pandas as pd
import traceback
import random
from typing import Literal

from strategies.temporal_difference_learning import TemporalDifferenceLearning
from utils.data_loaders import get_positions, get_5m_candles
from utils.orders import market_order
from utils.constants import CURRENCY_PAIRS, CURRENCY_PAIRS, POSITION_SIZE
from utils.traders.base_trader import BaseTrader

ActionType = Literal['buy', 'sell', 'hold']


class ExpectedSarsaTrader(TemporalDifferenceLearning):
    def __init__(self, trader, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        self.trader = trader
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = ['buy', 'sell', 'hold']  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        self.q_table = np.zeros((num_states, len(self.actions)))  # Q-table initialization

    def choose_action(self, state, min_price, max_price) -> ActionType:
        # Epsilon-greedy strategy for action selection
        state_idx = self.get_state_index(state, min_price, max_price)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            return self.actions[np.argmax(self.q_table[state_idx])]  # Exploitation

    def get_policy_probabilities(self, state):
        num_actions = len(self.q_table[state])
        policy = np.ones(num_actions) * (self.epsilon / num_actions)
        best_action = np.argmax(self.q_table[state])
        policy[best_action] += (1 - self.epsilon)
        return policy

    def update_q_table(self, state, action, reward, next_state, min_price, max_price):
        state_idx = self.get_state_index(state, min_price, max_price)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state, min_price, max_price)

        # Compute the expected Q-value for next_state using policy probabilities
        action_probs = self.get_policy_probabilities(next_state)
        expected_q_value = np.sum(action_probs * self.q_table[next_state_idx])

        # Q-table update formula
        self.q_table[state_idx, action_idx] = (1 - self.alpha) * self.q_table[state_idx, action_idx] + \
                                              self.alpha * (reward + self.gamma * expected_q_value)

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

            # Update Q-table
            self.update_q_table(current_price, action, reward, next_price, min_price, max_price)


def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame, trader: BaseTrader) -> None:
    # We want to create a new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = ExpectedSarsaTrader(trader)
    agent.train(data_for_currency)

    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')

    agent.perform_trading(data_for_currency, currency)


def expected_sarsa(trader):
    try:
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")

        for currency in CURRENCY_PAIRS:
            print(f"Running Expected SARSA for currency={currency}")

            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency, trader)

    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())
