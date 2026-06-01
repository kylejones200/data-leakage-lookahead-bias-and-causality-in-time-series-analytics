import logging

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


# --- FRED API Fetch ---
def fetch_fred_data(series_id, start_date="2000-01-01", end_date=None):
    """Fetch data from FRED via pandas_datareader."""
    from datetime import datetime

    import pandas_datareader.data as web

    if end_date is None:
        end_date = datetime.now()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    raw = web.DataReader(series_id, "fred", start, end)
    df = raw.reset_index()
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[-1]
    out = df.rename(columns={date_col: "date", value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def create_features_proper(df):
    """Create features properly without lookahead bias"""
    df = df.copy()
    df["next_day_price"] = df["value"].shift(-1)
    df["past_5day_ma"] = df["value"].rolling(window=5).mean()
    df["past_volatility"] = df["value"].rolling(window=10).std()
    return df


# --- Granger Causality ---
def granger_causality(data, max_lag=12):
    """Test pairwise Granger causality for all variable pairs in the dataframe"""
    results = {}
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                test_result = grangercausalitytests(
                    data[[col1, col2]].dropna(), maxlag=max_lag, verbose=False
                )
                min_p_value = min(
                    [test_result[i + 1][0]["ssr_ftest"][1] for i in range(max_lag)]
                )
                results[f"{col1} -> {col2}"] = min_p_value
    return results


# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your real FRED API key
    series_id = "PNGASJPUSDM"  # Japan LNG import price in USD
    df = fetch_fred_data_nixtla(series_id)
    df.set_index("date", inplace=True)
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
