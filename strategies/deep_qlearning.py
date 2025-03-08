import numpy as np
import random
from collections import deque
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import MetaTrader5 as mt5
import time

from utils.constants import CURRENCY_PAIRS

LOT_SIZE = 0.1
EPISODES = 1000
STEPS_PER_EPISODE = 500
BATCH_SIZE = 32

# ---------------------------
# Environment for Forex Trading
# ---------------------------
class ForexEnv:
    def __init__(self, symbol, lot_size):
        self.symbol = symbol
        self.lot_size = lot_size
        # Define state dimension (for example: [current price, moving average])
        self.state_dim = 2
        # Define actions: 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close positions
        self.action_space = [0, 1, 2, 3]
    
    def _get_market_data(self):
        # Get the latest tick data from MT5
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.last
        # For demonstration, compute a simple moving average over 10 ticks (you could replace this with any indicator)
        rates = mt5.copy_rates_from(self.symbol, mt5.TIMEFRAME_M1, time.time()-600, 10)
        ma = np.mean(rates['close']) if rates is not None else price
        return np.array([price, ma])
    
    def reset(self):
        # Reset the environment and return the initial state
        state = self._get_market_data()
        return state

    def step(self, action):
        """
        Perform an action in the environment.
        Action codes:
            0: Hold (do nothing)
            1: Buy (open a buy position)
            2: Sell (open a sell position)
            3: Close any open positions for the symbol
        """
        reward = 0
        done = False
        
        # Get current market data
        state = self._get_market_data()
        price = state[0]
        result = ''
        
        # Action execution:
        if action == 1:  # Buy
            # Create a buy order
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "DQN Buy Order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(order_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                reward = 0.1  # small positive reward for entering trade
            else:
                reward = -0.1  # penalty if order failed

        elif action == 2:  # Sell
            # Create a sell order
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "DQN Sell Order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(order_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                reward = 0.1
            else:
                reward = -0.1

        elif action == 3:  # Close open positions
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                reward = -0.05  # penalty for trying to close when no positions exist
            else:
                # Close each open position
                for pos in positions:
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        close_type = mt5.ORDER_TYPE_SELL
                    else:
                        close_type = mt5.ORDER_TYPE_BUY
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "DQN Close Order",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    close_result = mt5.order_send(close_request)
                    # You might want to compute reward based on profit/loss from this close action
                    if close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        reward += 0.2  # bonus for closing at a good moment
                    else:
                        reward -= 0.1
        else:
            # Hold action; perhaps incur a small time penalty to encourage action
            reward = -0.01

        print(f"order result={result}")

        # For a real application, you would compute reward based on P&L and risk metrics.
        # Also, define termination conditions (e.g., reaching a time limit or drawdown threshold).
        # Here we use a dummy condition.
        done = False

        # Get the next state
        next_state = self._get_market_data()
        return next_state, reward, done, {}

# ---------------------------
# DQN Agent using Keras
# ---------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95            # discount rate
        self.epsilon = 1.0           # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        # Experience replay: train the network using randomly sampled experiences
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_algorithm_for_currency(currency: str):    
    # Create environment and agent
    env = ForexEnv(currency, LOT_SIZE)
    print(f"Enironment created={env}")
    
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    print(f"Agent created={agent}")

    for e in range(EPISODES):
        state = env.reset()
        print(f"State reseted={state}")
        
        state = np.reshape(state, [1, state_size])
        for time_step in range(STEPS_PER_EPISODE):
            print(f"running action")
            
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Optionally, add sleep to pace your orders and avoid overloading MT5
            time.sleep(1)

            if done:
                print(f"Episode {e}/{EPISODES} ended at timestep {time_step}")
                break

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        # Print progress every episode
        print(f"Episode {e} completed, current exploration rate: {agent.epsilon:.2f}")

def deep_qlearning():
    for currency in CURRENCY_PAIRS:
        print(f"Running deep-qlearning alogirthm for currency={currency}")
        run_algorithm_for_currency(currency)
