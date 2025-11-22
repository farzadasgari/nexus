import numpy as np
import pandas as pd
from scipy import stats

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

Nd_list = [15, 30, 45, 60, 90]
for Nd in Nd_list: data = sp(data, Nd)

data['year'] = data.index.year
data['doy'] = data.index.dayofyear

Nd = 15
sp_col = f'SP_{Nd}'

qi_dict = {}
for doy, group in data.groupby('doy'):
    values = group[sp_col].dropna().values
    if len(values) == 0:
        continue
    M = group['year'].nunique()
    Zi = np.sum(values == 0)
    qi = Zi / M
    qi_dict[doy] = np.around(qi, 4)

DIST_NAMES = [
    'norm', # normal
    'expon', # exponential
    'gamma', # gamma
    'genextreme', # GEV
    'invgauss', # inverse Gaussian
    'logistic', # logistic
    'fisk', # log-logistic
    'lognorm', # log-normal
    'burr', # Burr
    'gumbel_r' # EV (extreme value)
]

AVAILABLE_DISTS = [name for name in DIST_NAMES if hasattr(stats, name)]
print("Using distributions:", AVAILABLE_DISTS)

def fit_distributions_per_doy(data, Nd=15, min_pos_values=10):
    sp_col = f'SP_{Nd}'
    day_params = {}
    for doy, group in data.groupby('doy'):
        values = group[sp_col].dropna().values
        if len(values) == 0: continue
        M = group['year'].nunique()
        Zi = np.sum(values == 0)
        qi = Zi / M
        pos_values = values[values > 0]
        if len(pos_values) < min_pos_values: continue
        best_aic = np.inf
        best_name = None
        best_params = None
        for name in AVAILABLE_DISTS:
            dist = getattr(stats, name)
            try:
                params = dist.fit(pos_values)
                loglik = np.sum(dist.logpdf(pos_values, *params))
                k = len(params)
                aic = 2 * k - 2 * loglik
                if aic < best_aic:
                    best_aic = aic
                    best_name = name
                    best_params = params
            except Exception:
                continue
        if best_name is None:
            continue
        day_params[doy] = {
            'qi': qi,
            'dist_name': best_name,
            'params': best_params
        }
    
    return day_params


def spi(data, Nd, day_params):
    sp_col = f'SP_{Nd}'
    spi_col = f'SPI_{Nd}'
    data[spi_col] = np.nan
    for doy, params in day_params.items():
        qi = params['qi']
        dist = getattr(stats, params['dist_name'])
        dist_params = params['params']
        mask = (data['doy'] == doy)
        sp_vals = data.loc[mask, sp_col].values
        if len(sp_vals) == 0:
            continue
        
        Fd = np.full_like(sp_vals, np.nan, dtype=float)
        is_zero = (sp_vals == 0)
        Fd[is_zero] = qi
        is_pos = (sp_vals > 0)
        if np.any(is_pos):
            Fi_vals = dist.cdf(sp_vals[is_pos], *dist_params)
            Fd[is_pos] = qi + (1 - qi) * Fi_vals
        
        Fd = np.clip(Fd, 1e-10, 1 - 1e-10)
        spi_vals = stats.norm.ppf(Fd)
        data.loc[mask, spi_col] = spi_vals
    return data


for Nd in Nd_list:
    day_params = fit_distributions_per_doy(data, Nd=Nd)
    data = compute_spi_for_Nd(data, Nd=Nd, day_params=day_params)
    data.to_csv("../dataset/cdhw_spi.csv")
