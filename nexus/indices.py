import pandas as pd
import numpy as np

extract = False
indices = True

if extract:
    input_path  = "../dataset/final_daily_all.csv"
    output_path = "../dataset/heatout.csv"

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
    print(f"\nSaved to: {output_path}")

if indices:
    df = pd.read_csv("../dataset/heatout.csv", low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    df['Tmean'] = (df['TMAX'] + df['TMIN']) / 2.0
    T95 = df['Tmean'].quantile(0.95)

    df['T3']   = df['Tmean'].rolling(window=3,  min_periods=3).mean()
    df['T30']  = df['Tmean'].rolling(window=30, min_periods=30).mean()

    df['EHIsig']  = df['T3'] - T95
    df['EHIaccl'] = df['T3'] - df['T30']
    df['EHF']     = df['EHIsig'] * np.maximum(1.0, df['EHIaccl'])

    LAT_DEG = 39.5177
    JULIAN_DAY = df['Date'].dt.dayofyear

    def thornthwaite_pet_safe(df_daily):
        tmean = df_daily['Tmean'].values

        monthly_t = df_daily['Tmean'].groupby(df_daily['Date'].dt.month).mean()
        I = sum(max(t, 0) / 5.0 ** 1.514 for t in monthly_t)
        if I <= 0: I = 1e-6
        a = 6.75e-7*I**3 - 7.71e-5*I**2 + 1.792e-2*I + 0.49239
        phi = np.radians(LAT_DEG)
        delta = 0.4093 * np.sin(2 * np.pi * JULIAN_DAY / 365 - 1.405)
        cos_arg = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
        omega = np.arccos(cos_arg)
        daylength = 24 * omega / np.pi
        ratio = 10 * tmean / I
        ratio = np.where(ratio < 0, 0, ratio)
        pet = 16 * (ratio ** a) * (daylength / 12)
        pet = np.where(tmean <= 0, 0.0, pet)
        return pet

    df['PET'] = thornthwaite_pet_safe(df)

    monthly = df.resample('MS', on='Date').agg({
        'PRCP': 'sum',
        'PET': 'sum'
    })
    monthly['BAL'] = monthly['PRCP'] - monthly['PET']

    ref_start = '1953-01-01'
    ref_end = '2025-05-01'
    bal_ref = monthly.loc[ref_start:ref_end, 'BAL']
    mu_ref = bal_ref.mean()
    sigma_ref = bal_ref.std()

    if sigma_ref == 0:
        raise ValueError("Zero standard deviation in reference period!")

    monthly['SPEI'] = (monthly['BAL'] - mu_ref) / sigma_ref

    month_start = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_aligned = monthly.reindex(month_start, method='ffill')

    df['PRCP_month'] = monthly_aligned['PRCP'].values
    df['PET_month'] = monthly_aligned['PET'].values
    df['BAL_month'] = monthly_aligned['BAL'].values
    df['SPEI'] = monthly_aligned['SPEI'].values

    df.to_csv("../dataset/heatout.csv", index=False)
