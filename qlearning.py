import traceback
import numpy as np
import random
import pandas as pd
import MetaTrader5 as mt5
import datetime as dt
import yfinance as yf

from utils.data_loaders import get_positions,get_5m_candles, get_positions_historical

class QLearningTrader:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1, num_states=100):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = actions  # Buy, Sell, Hold
        self.num_states = num_states  # Number of states (can be price ranges or features)
        self.q_table = np.zeros((num_states, len(actions)))  # Q-table initialization

    def get_state_index(self, state, min_price, max_price):
        # Normalize price to get a state index (for simplicity, using price-to-index mapping)
        normalized_state = (state - min_price) / (max_price - min_price)  # Normalize the price
        # Map to state index range [0, num_states - 1]
        return int(normalized_state * (self.num_states - 1))  # Ensure index is within bounds

    def get_action_index(self, action):
        return self.actions.index(action)

    def choose_action(self, state, min_price, max_price):
        # Epsilon-greedy strategy for action selection
        state_idx = self.get_state_index(state, min_price, max_price)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            return self.actions[np.argmax(self.q_table[state_idx])]  # Exploitation

    def update_q_table(self, state, action, reward, next_state, min_price, max_price):
        state_idx = self.get_state_index(state, min_price, max_price)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state, min_price, max_price)

        # Q-value update formula
        best_future_q = np.max(self.q_table[next_state_idx])  # Best Q-value for next state
        self.q_table[state_idx, action_idx] = (1 - self.alpha) * self.q_table[state_idx, action_idx] + \
                                                self.alpha * (reward + self.gamma * best_future_q)

    def calculate_reward(self, current_price, action, future_price):
        # Define the reward function for Forex trading
        if action == 'Buy':
            reward = future_price - current_price  # Price change after buying
        elif action == 'Sell':
            reward = current_price - future_price  # Price change after selling
        else:  # Hold
            reward = 0  # No reward for holding
        return reward

    def extract_data_from_positions(self, pos: pd.DataFrame):
        return pos['price_current'].values
    
    def get_min_and_max_price_from_data(self, data: np.ndarray) -> tuple[float, float]:
        min_price = np.min(data)
        max_price = np.max(data)
        
        return (min_price, max_price)

    def train(self, pos: pd.DataFrame) -> None:
        data = self.extract_data_from_positions(pos)
        
        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(data)
        
        print(f"min price: {min_price}")

        # Training on Forex data
        for i in range(1, len(data)):
            current_price = data[i-1]
            next_price = data[i]

            # Choose action using epsilon-greedy strategy
            action = self.choose_action(current_price, min_price, max_price)

            # Calculate the reward
            reward = self.calculate_reward(current_price, action, next_price)

            # Update Q-table
            self.update_q_table(current_price, action, reward, next_price, min_price, max_price)

    def test(self, pos: pd.DataFrame):
        data = self.extract_data_from_positions(pos)
        
        # Find min and max prices for normalization
        min_price, max_price = self.get_min_and_max_price_from_data(data)

        # Test the trained model with new data
        total_profit = 0
        for i in range(1, len(data)):
            current_price = data[i-1]
            action = self.choose_action(current_price, min_price, max_price)
            future_price = data[i]
            reward = self.calculate_reward(current_price, action, future_price)
            total_profit += reward
        return total_profit

def qlearning():    
    try:
        actions = ['Buy', 'Sell', 'Hold']
        
        symbols = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCNH']
        
        # results_per_symbol = {}
        
        agent = QLearningTrader(actions)
            
        open_pos = get_positions()
        print(f"open_pos head: {open_pos.head()}")
            
        agent.train(open_pos)
        
        print("Testing the trained model...")
        profit = agent.test(open_pos)
        print(f'Total profit from testing: {profit}')
            
    except Exception as e:
        print(f"Unexpected error while trying to perform q-learning algorithm: {e}")
        print(traceback.format_exc())

# # Fetch historical Forex data (EUR/USD) from Yahoo Finance
# # We use 'EURUSD=X' for the EUR/USD currency pair

# # data = yf.download('EURUSD=X', start='2020-01-01', end='2021-01-01')['Close'].values

# # Display the first few data points
# print("Forex Data (EUR/USD) - Close Prices:\n", data[:10])

# # Define possible actions: Buy, Sell, Hold
# actions = ['Buy', 'Sell', 'Hold']

# # Initialize the Q-learning agent
# agent = QLearningTrader(actions)

# # Train the agent with the Forex data
# print("Training the model...")
# agent.train(data)

# # After training, test the model to calculate the profit/loss
# print("Testing the trained model...")
# profit = agent.test(data)
# print(f'Total profit from testing: {profit}')
