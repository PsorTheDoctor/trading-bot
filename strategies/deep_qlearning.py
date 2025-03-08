import traceback
import numpy as np
import random
from collections import deque
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import MetaTrader5 as mt5
import time

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
    
    # def update_q_table(self, state, action: TradeAction, reward, next_state, min_price, max_price):
    #     print(f"state={state}")
    #     x = np.reshape(state, [1, self.state_size])
    #     predicted_model = self.model.predict(x, verbose=0)
    #     print(f"predicted model={predicted_model}")
        
    #     return predicted_model
    
    def fill_q_table(self, prices):
        # print(f"prices={prices}")
        # # x = np.reshape(prices, [1, self.state_size])
        # # print(f"x={x}")
        # predicted_model = self.model.predict(prices, verbose=0)
        # print(f"predicted model={predicted_model}")
        
        # return predicted_model
        
        # Create an empty Q-table with the right shape
        q_table = np.zeros((len(prices), self.action_size))
        # For each price (state), use the model to predict Q-values
        for i, price in enumerate(prices):
            # Reshape the price into the expected input shape (batch size 1, 1 feature)
            state = np.array([[price]])
            q_values = self.model.predict(state, verbose=0)
            q_table[i] = q_values
        self.q_table = q_table  # Save the Q-table in the class field
        

# # ---------------------------
# # DQN Agent using Keras
# # ---------------------------
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95            # discount rate
#         self.epsilon = 1.0           # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Network for Deep Q-Learning using Keras
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         # Epsilon-greedy action selection
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         q_values = self.model.predict(state, verbose=0)
#         return np.argmax(q_values[0])

#     def replay(self, batch_size):
#         # Experience replay: train the network using randomly sampled experiences
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
#             target_f = self.model.predict(state, verbose=0)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         # Decay exploration rate
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# def run_algorithm_for_currency(currency: str):    
#     # Create environment and agent
#     env = ForexEnv(currency, LOT_SIZE)
#     print(f"Enironment created={env}")
    
#     state_size = env.state_dim
#     action_size = len(env.action_space)
#     agent = DQNAgent(state_size, action_size)
#     print(f"Agent created={agent}")

#     for e in range(EPISODES):
#         state = env.reset()
#         print(f"State reseted={state}")
        
#         state = np.reshape(state, [1, state_size])
#         for time_step in range(STEPS_PER_EPISODE):
#             print(f"running action")
            
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state

#             # Optionally, add sleep to pace your orders and avoid overloading MT5
#             time.sleep(1)

#             if done:
#                 print(f"Episode {e}/{EPISODES} ended at timestep {time_step}")
#                 break

#             if len(agent.memory) > BATCH_SIZE:
#                 agent.replay(BATCH_SIZE)

#         # Print progress every episode
#         print(f"Episode {e} completed, current exploration rate: {agent.epsilon:.2f}")

def run_algorithm_for_currency(currency: str, data_for_currency: pd.DataFrame) -> None:
    # We want to create new agent for each currency, so training on data related to one currency won't impact buy/sell decisions for another
    agent = DeepQLearningTrader()
            
    agent.train(data_for_currency)
        
    print(f"Testing the trained model for currency={currency} ...")
    profit = agent.test(data_for_currency)
    print(f'Total profit from testing:{profit} for currency={currency}')
            
    agent.perform_trading(data_for_currency, currency)

def deep_qlearning():
    # for currency in CURRENCY_PAIRS:
    
    #     print(f"Running deep-qlearning alogirthm for currency={currency}")
    #     run_algorithm_for_currency(currency)
    
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
