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
    averaged_temps = (feb28[['TMAX', 'TMIN']] + feb29[['TMAX', 'TMIN']]) / 2
    summed_prcp = feb28['PRCP'] + feb29['PRCP']
    data.loc[f"{year}-02-28", 'PRCP'] = summed_prcp
    data.loc[f"{year}-02-28", ['TMAX', 'TMIN']] = averaged_temps
    
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
    if len(values) == 0: continue
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
            except Exception: continue
        if best_name is None: continue
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
        if len(sp_vals) == 0: continue
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
    data = spi(data, Nd=Nd, day_params=day_params)


def tacc(data, Nh=3):
    col_name = f'TACC_{Nh}'
    data[col_name] = (
        data['TMEAN']
        .rolling(window=Nh, min_periods=Nh)
        .mean()
    )
    return data


Nh_list = [3, 5, 7, 10, 15]
for Nh in Nh_list:
    data = tacc(data, Nh)


def shi(data, Nh, min_sample=10):
    Tacc_col = f'TACC_{Nh}'
    SHI_col = f'SHI_{Nh}'
    data[SHI_col] = np.nan
    years = np.sort(data['year'].unique())
    n_years = len(years)
    year_to_idx = {y: i for i, y in enumerate(years)}
    for doy in sorted(data['doy'].unique()):
        mask_doy = (data['doy'] == doy)
        for year in years:
            mask_row = mask_doy & (data['year'] == year)
            if not mask_row.any(): continue
            idx_row = data.index[mask_row][0]
            T_hat = data.at[idx_row, Tacc_col]
            if pd.isna(T_hat): continue
            yi = year_to_idx[year]
            if n_years >= 30:
                if yi >= 29:
                    window_years = years[yi-29:yi+1]
                else:
                    window_years = years[:30]
            else:
                window_years = years
            mask_sample = mask_doy & data['year'].isin(window_years)
            sample_vals = data.loc[mask_sample, Tacc_col].dropna().values
            if len(sample_vals) < min_sample: continue  
            best_aic = np.inf
            best_name = None
            best_params = None
            for name in AVAILABLE_DISTS:
                dist = getattr(stats, name)
                try:
                    params = dist.fit(sample_vals)
                    loglik = np.sum(dist.logpdf(sample_vals, *params))
                    k = len(params)
                    aic = 2 * k - 2 * loglik
                    if aic < best_aic:
                        best_aic = aic
                        best_name = name
                        best_params = params
                except Exception: continue
            if best_name is None: continue
            dist = getattr(stats, best_name)
            Fi = dist.cdf(T_hat, *best_params)
            Fi = np.clip(Fi, 1e-10, 1 - 1e-10)
            SHI_val = stats.norm.ppf(Fi)
            data.at[idx_row, SHI_col] = SHI_val
    return data


for Nh in Nh_list:
    data = shi(data, Nh=Nh)
    data.to_csv("../dataset/cdhw.csv")
