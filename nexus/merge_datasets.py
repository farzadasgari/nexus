import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from config import * 

cdec_path = cdec_save_path
noaa_path = noaa_save_path
era5_path = era_mean_save_path
output_field_path = merged_dataset_path_field
output_all_path = merged_dataset_path_all

df_cdec = pd.read_csv(cdec_path)
df_cdec['Date'] = pd.to_datetime(df_cdec['OBS DATE'], format='%Y%m%d %H%M').dt.normalize()
df_cdec['VALUE'] = pd.to_numeric(df_cdec['VALUE'], errors='coerce')
df_cdec['sensor_col'] = df_cdec['SENSOR_TYPE'].str.replace(' ', '_') + '_' + df_cdec['SENSOR_NUMBER'].astype(str)
df_cdec_pivoted = df_cdec.pivot_table(
    index='Date',
    columns='sensor_col',
    values='VALUE',
    aggfunc='mean'
).reset_index()

units_df = df_cdec[['SENSOR_NUMBER', 'SENSOR_TYPE', 'UNITS']].drop_duplicates().set_index('SENSOR_TYPE')

df_noaa = pd.read_csv(noaa_path)
df_noaa['Date'] = pd.to_datetime(df_noaa['DATE']).dt.normalize()

df_merged = pd.merge(df_cdec_pivoted, df_noaa.drop('DATE', axis=1), on='Date', how='outer')
os.makedirs(os.path.dirname(output_all_path), exist_ok=True)
df_merged['Date'] = df_merged['Date'].dt.date
df_merged.to_csv(output_field_path, index=False)

df_era = pd.read_csv(era5_path)
df_era['Date'] = pd.to_datetime(df_era['Date']).dt.normalize()
df_merged = pd.merge(df_cdec_pivoted, df_noaa.drop('DATE', axis=1), on='Date', how='outer')
df_merged = pd.merge(df_merged, df_era, on='Date', how='outer')
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

df_merged['Date'] = df_merged['Date'].dt.date

df_merged.to_csv(output_all_path, index=False)

