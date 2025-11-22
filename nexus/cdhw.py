import numpy as np
import pandas as pd

data = pd.read_csv("../dataset/climate_features.csv", low_memory=False)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

for col in ['PRCP', 'TMAX', 'TMIN']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

leap_years = data.index[data.index.is_leap_year].year.unique()
for year in leap_years:
    feb28 = data.loc[f"{year}-02-28"]
    feb29 = data.loc[f"{year}-02-29"]
    averaged = (feb28[['PRCP', 'TMAX', 'TMIN']] + feb29[['PRCP', 'TMAX', 'TMIN']]) / 2
    data.loc[f"{year}-02-28", ['PRCP', 'TMAX', 'TMIN']] = averaged
data = data[~((data.index.month == 2) & (data.index.day == 29))]

data['TMEAN'] = data[['TMAX', 'TMIN']].mean(axis=1)

def sp(data, Nd=15):
    data[f'SP_{Nd}'] = (
        data['PRCP']
        .rolling(window=Nd, min_periods=Nd)
        .sum()
    )
    return data

for Nd in [15, 30, 45, 60, 90]: data = sp(data, Nd)
    
