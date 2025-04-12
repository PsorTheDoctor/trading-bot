from pathlib import Path
from os import path, makedirs
import numpy as np
import random
import pandas as pd
from abc import ABC, abstractmethod

from utils.orders import market_order
from utils.constants import POSITION_SIZE, TradeAction

MAX_TRADES_PER_ALGORITHM_ITERATION = 5

Q_TABLE_FILE_DIRECTORY_PATH = path.join(Path(__file__ ).parent.parent, 'cache')
Q_TABLE_FILE_NAME = 'qlearning-qtable-content.npy'
Q_TABLE_FILE_FULL_PATH = path.join(Q_TABLE_FILE_DIRECTORY_PATH, Q_TABLE_FILE_NAME)

class BaseQLearningTrader(ABC):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):        
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD, TradeAction.CLOSE_POSITIONS]  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        
        self.load_q_table_from_file()

    def load_q_table_from_file(self):
        print('Loading q-table content from file...')
        
        try:
            self.q_table = np.load(Q_TABLE_FILE_FULL_PATH)
        except FileNotFoundError:
            print('Could not load q-table content from file - fallback to default value...')
            self.q_table = np.zeros((self.num_states, len(self.actions)))  # Q-table initialization

    def save_q_table_in_file(self):
        print('Saving q-table content to file...')
        makedirs(Q_TABLE_FILE_DIRECTORY_PATH, exist_ok=True)
        np.save(Q_TABLE_FILE_FULL_PATH, self.q_table)

    def get_state_index(self, state, min_price, max_price):
        # Normalize price to get a state index (for simplicity, using price-to-index mapping)
        normalized_state = (state - min_price) / (max_price - min_price)  # Normalize the price
        # Map to state index range [0, num_states - 1]
        return int(normalized_state * (self.num_states - 1))  # Ensure index is within bounds

    def get_action_index(self, action):
        return self.actions.index(action)

    def choose_action(self, state, min_price, max_price) -> TradeAction:
        # Epsilon-greedy strategy for action selection
        state_idx = self.get_state_index(state, min_price, max_price)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            return self.actions[np.argmax(self.q_table[state_idx])]  # Exploitation
    
    @abstractmethod
    def fill_q_table(self, prices: list[float]):
        pass

    def calculate_reward(self, current_price: float, action: TradeAction, future_price: float) -> float:
        # Define the reward function for Forex trading
        if action == TradeAction.BUY:
            reward = future_price - current_price  # Price change after buying
        elif action == TradeAction.SELL:
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

        # Training on Forex data
        self.fill_q_table(prices)
        
        self.save_q_table_in_file()

    def test(self, data: pd.DataFrame) -> float:
        prices = self.extract_prices_from_data(data)
        
        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(prices)

        # Test the trained model with new data
        total_profit = 0
        for i in range(1, len(prices)):
            current_price = prices[i-1]
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
        max_actions_per_algorithm_iteration = min(MAX_TRADES_PER_ALGORITHM_ITERATION, len(prices))
        for i in range(1, max_actions_per_algorithm_iteration):
            current_price = prices[i-1]
            action = self.choose_action(current_price, min_price, max_price)
            future_price = prices[i]
            
            print(f"action: {action}")
            
            if action == TradeAction.BUY or action == TradeAction.SELL:
                order_status = market_order(currency, POSITION_SIZE, action.value)
                print(f"order status: {order_status}")
