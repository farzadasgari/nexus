import numpy as np
import pandas as pd
from scipy.stats import gamma

def load_data(file_path):
    data = pd.read_csv(file_path, low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def spi(data, window_size=3, spi_threshold=-1):
    data['precip_cum'] = data['PRCP'].rolling(window=window_size).sum()
    shape, loc, scale = gamma.fit(data['precip_cum'].dropna())
    data['SPI'] = (data['precip_cum'] - np.mean(data['precip_cum'])) / np.std(data['precip_cum'])
    # data['SPI_Drought'] = (data['SPI'] < spi_threshold).astype(int)
    return data.drop(['precip_cum'], axis=1)


def pet(data, LAT_DEG=39.5177):
    JULIAN_DAY = data['Date'].dt.dayofyear
    data['Tmean'] = (data['TMAX'] + data['TMIN']) / 2
    tmean = data['Tmean'].values
    monthly_t = data['Tmean'].groupby(data['Date'].dt.month).mean()
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
    data["PET"] = pet
    return data.drop(['Tmean'], axis=1)

    
def daily_spei(data, window_size=3, spei_threshold=-1):
    data['PET_diff'] = data['PRCP'] - data['PET']
    data['SPEI'] = (data['PET_diff'] - np.mean(data['PET_diff'])) / np.std(data['PET_diff'])
    # data['SPEI_Drought'] = (data['SPEI'] < spei_threshold).astype(int)
    return data.drop(['PET_diff'], axis=1)


def pwm_lmoments(x):
    x = np.sort(x)
    n = len(x)
    b0 = np.mean(x)
    b1 = np.sum([(i/(n-1))*x[i] for i in range(n)]) / n
    b2 = np.sum([(i*(i-1))/((n-1)*(n-2))*x[i] for i in range(n)]) / n if n > 2 else 0
    l1 = b0
    l2 = 2*b1 - b0
    l3 = 6*b2 - 6*b1 + b0
    t3 = l3/l2 if l2 != 0 else 0
    return l1, l2, t3


def fit_log_logistic(x):
    x = x[~np.isnan(x)]
    l1, l2, t3 = pwm_lmoments(x)
    beta = (0.2906 - 0.1882 * t3 + 0.0442 * t3**2) / (1 - t3)
    alpha = l2 * beta / np.pi
    gamma = l1 - alpha * np.tan(np.pi/(2*beta))
    return alpha, beta, gamma


def loglogistic_cdf(x, alpha, beta, gamma):
    z = (x - gamma) / alpha
    z = np.maximum(z, 1e-12)
    return 1.0 / (1.0 + z**(-beta))


def spei_from_cdf(F):
    P = 1 - F
    P = np.clip(P, 1e-12, 1 - 1e-12)
    C0, C1, C2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    W = np.sqrt(-2.0 * np.log(P))
    spei = W - (C0 + C1*W + C2*W*W) / (1 + d1*W + d2*W*W + d3*W*W*W)
    spei[P > 0.5] *= -1
    return spei


def compute_spei_series(D_series, scale):
    Dk = D_series.rolling(scale).sum()
    valid = Dk.dropna().values
    if len(valid) < 30:
        return pd.Series(np.full_like(Dk, np.nan), index=Dk.index)
    alpha, beta, gamma = fit_log_logistic(valid)
    F = loglogistic_cdf(Dk.values, alpha, beta, gamma)
    spei_vals = spei_from_cdf(F)
    return pd.Series(spei_vals, index=Dk.index)


def periodic_spei(data):
    df = data.copy()
    df["D"] = df["PRCP"] - df["PET"]
    weekly = df.resample("W-SUN", on="Date").sum()
    weekly["D"] = weekly["PRCP"] - weekly["PET"]
    weekly["SPEI_1w"] = compute_spei_series(weekly["D"], scale=1)
    monthly = df.resample("MS", on="Date").sum()
    monthly["D"] = monthly["PRCP"] - monthly["PET"]
    for k in [1, 3, 6, 12]:
        monthly[f"SPEI_{k}m"] = compute_spei_series(monthly["D"], scale=k)
    return weekly, monthly


df = load_data("../dataset/climate_features.csv")
df = pet(df)
df = spi(df)
df = daily_spei(df)
w, m = periodic_spei(df)
df = df.set_index("Date")
w_daily = w.reindex(df.index, method="ffill")
m_daily = m.reindex(df.index, method="ffill")
w_cols = [c for c in w_daily.columns if "SPEI" in c.upper()]
m_cols = [c for c in m_daily.columns if "SPEI" in c.upper()]
df[w_cols] = w_daily[w_cols]
df[m_cols] = m_daily[m_cols]
df.to_csv("../dataset/droughts.csv")
