import pandas as pd
import numpy as np

input_path  = "../dataset/final_daily_all.csv"
output_path = "../dataset/climate_features.csv"

df = pd.read_csv(input_path, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

start_date = pd.to_datetime('1953-04-22')
end_date = pd.to_datetime('2025-05-11')

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

missing_mask_prcp = df['PRCP'].isna()
missing_mask_tmax = df['TMAX'].isna()
missing_mask_tmin = df['TMIN'].isna()

if 'total_precipitation_sum' in df.columns:
    df.loc[missing_mask_prcp, 'PRCP'] = df.loc[missing_mask_prcp, 'total_precipitation_sum']
if 'temperature_2m_max_C' in df.columns:
    df.loc[missing_mask_tmax, 'TMAX'] = df.loc[missing_mask_tmax, 'temperature_2m_max_C']
if 'temperature_2m_min_C' in df.columns:
    df.loc[missing_mask_tmin, 'TMIN'] = df.loc[missing_mask_tmin, 'temperature_2m_min_C']

core_columns = ['Date', 'PRCP', 'TMAX', 'TMIN']
df_clean = df[core_columns].copy()

df_clean.to_csv(output_path, index=False)
