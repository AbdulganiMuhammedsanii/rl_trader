# gen_prices.py
import pandas as pd
import numpy as np

np.random.seed(0)
prices = 100 + np.cumsum(np.random.randn(300))  # 300 timesteps of random walk
pd.DataFrame({"close": prices}).to_csv("price_series.csv", index=False)
