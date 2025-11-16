import numpy as np
import pandas as pd

def unit_convertor(old_path, new_path):
    df = pd.read_csv(old_path, parse_dates=['Date'])

    if 'STORAGE_15' in df.columns:
        df['storage_m3'] = (df['STORAGE_15'] * 1233.481837).round(2)
        df.drop('STORAGE_15', axis=1, inplace=True)

    if 'RES_ELE_6' in df.columns:
        df['elevation_m'] = (df['RES_ELE_6'] * 0.3048).round(2)
        df.drop('RES_ELE_6', axis=1, inplace=True)

    flow_cols = ['INFLOW_76', 'OUTFLOW_23', 'RIV_REL_85']
    for col in flow_cols:
        if col in df.columns:
            df[col + '_m3s'] = (df[col] * 0.0283168).round(2)
            df.drop(col, axis=1, inplace=True)

    precip_cols = ['PPT_INC_45', 'PPTINC4_203', 'RAIN_2']
    for col in precip_cols:
        if col in df.columns:
            df[col + '_mm'] = (df[col] * 25.4).round(2)
            df.drop(col, axis=1, inplace=True)

    if 'EVAP_74' in df.columns:
        df['evap_mm'] = (df['EVAP_74'] * 25.4).round(2)
        df.drop('EVAP_74', axis=1, inplace=True)

    temp_cols = [col for col in df.columns if 'temperature' in col.lower() or 'dewpoint' in col.lower()]
    for col in temp_cols:
        new_col = col + '_C' if not col.endswith('_C') else col
        df[new_col] = (df[col] - 273.15).round(2)
        if new_col != col:
            df.drop(col, axis=1, inplace=True)

    evap_cols = [col for col in df.columns if 'evaporation' in col.lower()]
    for col in evap_cols:
        df[col] = (-df[col] * 1000).round(2)
    prcp_cols = [col for col in df.columns if 'precipitation' in col.lower()]
    for col in prcp_cols:
        df[col] = (df[col] * 100).round(2)
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    df = df.sort_index()
    df.to_csv(new_path, index=False)

unit_convertor("../dataset/merged_daily_all.csv", "../dataset/final_daily_all.csv")
unit_convertor("../dataset/merged_daily_field.csv", "../dataset/final_daily_field.csv")
