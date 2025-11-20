import numpy as np
import pandas as pd
from scipy.stats import norm

def load_data(file_path):
    data = pd.read_csv(file_path, low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def constant_threshold(data, threshold=35, consecutive_days=3):
    data['Heatwave_Constant'] = (data['TMAX'] > threshold).astype(int)
    data['Heatwave_Constant'] = data['Heatwave_Constant'].rolling(window=consecutive_days).sum()
    data['Heatwave_Constant'] = (data['Heatwave_Constant'] >= consecutive_days).astype(int)
    return data

def pdf_threshold(data, window_size=15, percentile=90):
    data['Heatwave_Percentile'] = data['TMAX'].rolling(window=window_size, min_periods=1).apply(
        lambda x: np.percentile(x, percentile), raw=True
    )
    data['Heatwave_PDF'] = (data['TMAX'] > data['Heatwave_Percentile']).astype(int)
    return data.drop(['Heatwave_Percentile'], axis=1)

def calculate_shi(data, window=15):
    tmean = (data['TMAX'] + data['TMIN']) / 2
    rank = tmean.rank(pct=True)
    return norm.ppf(rank)

def shi(data, threshold_shi=1):
    data['SHI'] = calculate_shi(data)
    data['SHI_Heatwave'] = (data['SHI'] >= threshold_shi).astype(int)
    return data

def ehf(data, percentile=95, window_rolling=30, threshold_ehf=0):
    data['Tmean'] = (data['TMAX'] + data['TMIN']) / 2
    T95 = data['Tmean'].rolling(window=window_rolling, min_periods=1).apply(lambda x: np.percentile(x, percentile), raw=True)
    data['Tmean_3day_avg'] = data['Tmean'].rolling(window=3).mean()
    data['EHI_sig'] = data['Tmean_3day_avg'] - T95
    data['Tmean_30day_avg'] = data['Tmean'].rolling(window=30).mean()
    data['EHI_accl'] = data['Tmean_3day_avg'] - data['Tmean_30day_avg']
    data['EHF'] = data['EHI_sig'] * np.maximum(data['EHI_accl'], 1)
    data['EHF_Heatwave'] = (data['EHF'] >= threshold_ehf).astype(int)
    return data.drop(['Tmean', 'Tmean_3day_avg', 'Tmean_30day_avg', 'EHI_sig', 'EHI_accl'], axis=1)

df = load_data("../dataset/climate_features.csv")
df = constant_threshold(df)
df = pdf_threshold(df)
df = shi(df)
df = ehf(df)

df.to_csv("../dataset/heatwaves.csv", index=False)
