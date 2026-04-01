import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

warnings.filterwarnings('ignore')

# --- FRED API Fetch ---
def fetch_fred_data(series_id, api_key, start_date='2000-01-01'):
    """Fetch data from FRED API"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df
    else:
        raise Exception(f"FRED API Error {response.status_code}")

# --- Feature Engineering ---
def create_features(df, leakage=False):
    """Create features with or without data leakage"""
    df = df.copy()
    if leakage:
        df['rolling_mean'] = df['value'].rolling(window=7, center=False).shift(1).mean()
        df['volatility'] = df['value'].rolling(window=10, center=False).shift(1).std()
    else:
        df['rolling_mean'] = df['value'].rolling(window=7).mean().shift(1)
        df['volatility'] = df['value'].rolling(window=10).std().shift(1)
    df['price_lag'] = df['value'].shift(1)
    df['monthly_return'] = df['value'].pct_change(periods=30)
    return df

# --- Feature Engineering (Lookahead Bias) ---
def create_features_with_lookahead(df):
    """Create features improperly with lookahead bias"""
    df = df.copy()
    df['next_day_price'] = df['value'].shift(-1)  # target
    df['future_5day_ma'] = df['value'].rolling(window=5, center=False).shift(1).mean()
    df['future_volatility'] = df['value'].rolling(window=10, center=False).shift(1).std()
    return df

def create_features_proper(df):
    """Create features properly without lookahead bias"""
    df = df.copy()
    df['next_day_price'] = df['value'].shift(-1)
    df['past_5day_ma'] = df['value'].rolling(window=5).mean()
    df['past_volatility'] = df['value'].rolling(window=10).std()
    return df

# --- Granger Causality ---
def granger_causality(data, max_lag=12):
    """Test pairwise Granger causality for all variable pairs in the dataframe"""
    results = {}
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                test_result = grangercausalitytests(data[[col1, col2]].dropna(), maxlag=max_lag, verbose=False)
                min_p_value = min([test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)])
                results[f"{col1} -> {col2}"] = min_p_value
    return results

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your real FRED API key
    api_key = "YOUR_FRED_API_KEY"
    series_id = "PNGASJPUSDM"  # Japan LNG import price in USD

    df = fetch_fred_data(series_id, api_key)
    df.set_index('date', inplace=True)

    # WRONG: Lookahead leakage
    leaky_features = create_features(df, leakage=True)

    # RIGHT: No leakage
    clean_features = create_features(df, leakage=False)

    # Lookahead bias example
    df_lookahead = create_features_with_lookahead(df)
    df_proper = create_features_proper(df)

    # Granger causality test (example: create dummy data or use actual multivariate series)
    # dummy_df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
    # logging.info(granger_causality(dummy_df))
