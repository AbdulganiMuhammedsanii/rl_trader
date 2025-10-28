#trading environment class for reinforcement learning

import numpy as np
import pandas as pd

class TradingEnv:
  def __init__(self, csv_path, window_size = 30, gamma = 0.99):
    self.window_size = window_size
    self.gamma = gamma
    self.data = pd.read_csv(csv_path)

    # we assume the column is named 'Close'

    self.prices = self.data['Close'].values

    self.reset()



  def reset(self):
    self.t = self.window_size
    self.position_flag = 0 # 0 for flat, 1 for long

    self.entry_price = 0.0
    self.cash = 0.0

    state = self._get_state()
    return state
  
  def step(self, action):
    price = self.prices[self.t]

    reward = 0.0
    if action == 2 and self.position_flag == 1: # selling and was long
      trade_return = (price - self.entry_price) / self.entry_price
      self.cash += trade_return
      reward = trade_return
      self.position_flag = 0
      self.entry_price = 0.0
    elif action == 1 and self.position_flag == 0: # buying and was flat
      self.entry_price = price
      self.position_flag = 1


    # hold or invalid buy/sell combos default rewards below

    if self.position_flag == 1:
      unrealized = (price - self.entry_price) / self.entry_price
      reward += unrealized

    self.t += 1
    done = (self.t >= len(self.prices))  
    next_state = self._get_state()   
    return next_state, reward, done

  def _get_state(self):
    if self.t >= len(self.prices):
      t_idx = len(self.prices) - 1
    else:
      t_idx = self.t

    window_prices = self.prices[t_idx - self.window_size:t_idx]
    curr_price = self.prices[t_idx]

    #normalize window to current price

    norm_window = window_prices / curr_price

    if self.position_flag == 1:
      unreal = (curr_price - self.entry_price) / self.entry_price
    else:
      unreal = 0.0

    state = np.concatenate([norm_window, np.array([self.position_flag, unreal], dtype=np.float64)
    ])
    return state.astype(np.float32)
  def state_size(self):
    # window size + position flag + unrealized return
    return (self.window_size + 2)

  def num_actions(self):
    # 0 for hold, 1 for buy, 2 for sell
    return 3