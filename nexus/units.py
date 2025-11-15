import numpy as np
import pandas as pd

def unit_convertor(old_path, new_path):
    df = pd.read_csv(old_path, parse_dates=['Date'], index_col='Date')

    if 'STORAGE_15' in df.columns:
        df['storage_m3'] = df['STORAGE_15'] * 1233.481837
        df.drop('STORAGE_15', axis=1, inplace=True)

    if 'RES_ELE_6' in df.columns:
        df['elevation_m'] = df['RES_ELE_6'] * 0.3048
        df.drop('RES_ELE_6', axis=1, inplace=True)

    flow_cols = ['INFLOW_76', 'OUTFLOW_23', 'RIV_REL_85']
    for col in flow_cols:
        if col in df.columns:
            df[col + '_m3s'] = df[col] * 0.0283168
            df.drop(col, axis=1, inplace=True)

    precip_cols = ['PPT_INC_45', 'PPTINC4_203', 'RAIN_2']
    for col in precip_cols:
        if col in df.columns:
            df[col + '_mm'] = df[col] * 25.4
            df.drop(col, axis=1, inplace=True)

    if 'EVAP_74' in df.columns:
        df['evap_mm'] = df['EVAP_74'] * 25.4
        df.drop('EVAP_74', axis=1, inplace=True)

    kelvin_cols = ['temperature_2m', 'skin_temperature', 'lake_bottom_temperature', 'lake_mix_layer_temperature', 'lake_total_layer_temperature']
    for col in kelvin_cols:
        if col in df.columns:
            df[col + '_C'] = df[col] - 273.15

    df = df.sort_index()
    tmp = df.select_dtypes(include=[np.number])
    df.loc[:, tmp.columns] = np.round(tmp, 2)
    df.to_csv(new_path)

unit_convertor("../dataset/merged_daily_all.csv", "../dataset/final_daily_all.csv")
unit_convertor("../dataset/merged_daily_field.csv", "../dataset/final_daily_field.csv")
