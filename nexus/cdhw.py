import numpy as np
import pandas as pd

data = pd.read_csv("../dataset/climate_features.csv", low_memory=False)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


