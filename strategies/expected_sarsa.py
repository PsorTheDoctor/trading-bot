import numpy as np
import pandas as pd
import traceback
import random
from typing import Literal

from utils.data_loaders import get_positions, get_5m_candles
from utils.orders import market_order
from utils.constants import CURRENCY_PAIRS, CURRENCY_PAIRS, POSITION_SIZE

ActionType = Literal['buy', 'sell', 'hold']


class ExpectedSarsaTrader:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = ['buy', 'sell', 'hold']  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        self.q_table = np.zeros((num_states, len(self.actions)))  # Q-table initialization

    def get_state_index(self, state, min_price, max_price):
        # Normalize price to get a state index (for simplicity, using price-to-index mapping)
        normalized_state = (state - min_price) / (max_price - min_price)  # Normalize the price
        # Map to state index range [0, num_states - 1]
        return int(normalized_state * (self.num_states - 1))  # Ensure index is within bounds

    def get_action_index(self, action):
        return self.actions.index(action)

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

    def calculate_reward(self, current_price: float, action: ActionType, future_price: float) -> float:
        # Define the reward function for Forex trading
        if action == 'buy':
            reward = future_price - current_price  # Price change after buying
        elif action == 'sell':
            reward = current_price - future_price  # Price change after selling
        else:  # Hold
            reward = 0  # No reward for holding
        return reward

    def extract_prices_from_data(self, data: pd.DataFrame) -> list[float]:
        return data['close'].values

    def get_min_and_max_price_from_data(self, data: np.ndarray) -> tuple[float, float]:
        min_price = np.min(data)
        max_price = np.max(data)

        return (min_price, max_price)

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

    def test(self, data: pd.DataFrame) -> float:
        prices = self.extract_prices_from_data(data)

        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(prices)

        # Test the trained model with new data
        total_profit = 0
        for i in range(1, len(prices)):
            current_price = prices[i - 1]
            action = self.choose_action(current_price, min_price, max_price)
            future_price = prices[i]
            reward = self.calculate_reward(current_price, action, future_price)
            total_profit += reward
        return total_profit

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
                order_status = market_order(currency, POSITION_SIZE, action)
                print(f"order status: {order_status}")


def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame) -> None:
    # We want to create a new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = ExpectedSarsaTrader()
    agent.train(data_for_currency)

    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')

    agent.perform_trading(data_for_currency, currency)


def expected_sarsa():
    try:
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")

        for currency in CURRENCY_PAIRS:
            print(f"Running Expected SARSA for currency={currency}")

            data_for_currency = get_5m_candles(currency)
            print(f"data for currency ({currency})={data_for_currency.head()}")
            run_algorithm_for_currency(currency, data_for_currency)

    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())