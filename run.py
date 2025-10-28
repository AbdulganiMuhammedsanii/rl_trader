from env import TradingEnv
import matplotlib.pyplot as plt

#importing my TradingEnv class
import numpy as np

env = TradingEnv("price_series.csv", window_size=30)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.random.choice([0, 1, 2])  # random HOLD/BUY/SELL
    next_state, reward, done = env.step(action)
    total_reward += reward

print("Final cash (realized profit):", env.cash)
print("Total reward accumulated:", total_reward)
plt.plot(env.prices)
plt.title("Synthetic Price Series")
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.show()