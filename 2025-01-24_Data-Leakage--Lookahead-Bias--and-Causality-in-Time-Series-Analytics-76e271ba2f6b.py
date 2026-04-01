# Description: Short example for Data Leakage Lookahead Bias and Causality in Time Series Analytics.



from datetime import datetime
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


warnings.filterwarnings('ignore')


def fetch_fred_data(series_id, start_date='2000-01-01', end_date=None):
    """Fetch FRED data using pandas_datareader."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
    df = df.rename(columns={series_id: 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()
    df = df.reset_index()  
    df = df.rename(columns={'DATE': 'date'})  
    return df

def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_features(df, leakage=False):
    """Create features with or without data leakage"""
    df = df.copy()
    if leakage: #Leakage is bad
        df['rolling_mean'] = df['value'].rolling(window=7, center=True).mean()
        df['volatility'] = df['value'].rolling(window=10, center=True).std()
    else:
        df['rolling_mean'] = df['value'].rolling(window=7).mean().shift(1)
        df['volatility'] = df['value'].rolling(window=10).std().shift(1)
    
    df['price_lag'] = df['value'].shift(1)
    df['monthly_return'] = df['value'].pct_change(periods=30)
    return df

def train_model(data, features, target='value'):
    """Train and evaluate model"""
    data = data.dropna()
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return test_data.index, y_test, y_pred

def plot_features(data, leakage_data, proper_data, title, filename):
    """Plot feature comparison for leakage and proper handling"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rolling means
    ax1.plot(data.index, data['value'], label='Original Price', alpha=0.5)
    ax1.plot(leakage_data.index, leakage_data['rolling_mean'], 
             label='Rolling Mean (with leakage)', linewidth=2)
    ax1.plot(proper_data.index, proper_data['rolling_mean'], 
             label='Rolling Mean (proper)', linewidth=2)
    ax1.set_title(f'{title} - Rolling Means')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # Plot volatility
    ax2.plot(leakage_data.index, leakage_data['volatility'], 
             label='Volatility (with leakage)', linewidth=2)
    ax2.plot(proper_data.index, proper_data['volatility'], 
             label='Volatility (proper)', linewidth=2)
    ax2.set_title(f'{title} - Volatility')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(leakage_results, proper_results, title, filename):
    """Plot prediction results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Unpack results
    dates_leak, y_test_leak, y_pred_leak = leakage_results
    dates_proper, y_test_proper, y_pred_proper = proper_results
    
    # Calculate MAPE
    mape_leak = mape(y_test_leak, y_pred_leak)
    mape_proper = mape(y_test_proper, y_pred_proper)
    
    # Time series predictions
    ax1.plot(dates_leak, y_test_leak, label='Actual', alpha=0.7)
    ax1.plot(dates_leak, y_pred_leak, '--', label=f'With Leakage (MAPE: {mape_leak:.2f}%)')
    ax1.plot(dates_proper, y_pred_proper, '--', label=f'Proper (MAPE: {mape_proper:.2f}%)')
    ax1.set_title(f'{title} - Predictions Over Time')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    
    # Scatter plots
    ax2.scatter(y_test_leak, y_pred_leak, alpha=0.5, label='With Leakage')
    ax2.scatter(y_test_proper, y_pred_proper, alpha=0.5, label='Proper')
    ax2.plot([min(y_test_leak.min(), y_test_proper.min()), 
              max(y_test_leak.max(), y_test_proper.max())],
             [min(y_test_leak.min(), y_test_proper.min()),
              max(y_test_leak.max(), y_test_proper.max())],
             'r--', label='Perfect Prediction')
    ax2.set_title('Actual vs Predicted Prices')
    ax2.legend(loc='upper left')
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Fetch data
    japan_gas = fetch_fred_data('PNGASJPUSDM')
    
    # Data Leakage Analysis
    data_with_leakage = create_features(japan_gas, leakage=True)
    data_proper = create_features(japan_gas, leakage=False)
    
    # Plot feature comparison
    plot_features(japan_gas, data_with_leakage, data_proper, 
                 'Japan Natural Gas Prices', 'japan_gas_features.png')
    
    # Train models
    features_leak = ['rolling_mean', 'volatility', 'price_lag', 'monthly_return']
    features_proper = ['rolling_mean', 'volatility', 'price_lag', 'monthly_return']
    
    leakage_results = train_model(data_with_leakage, features_leak)
    proper_results = train_model(data_proper, features_proper)
    
    # Plot prediction results
    plot_predictions(leakage_results, proper_results,
                    'Japan Natural Gas Prices', 'japan_gas_predictions.png')
    
    # Calculate and print MAPE
    _, y_test_leak, y_pred_leak = leakage_results
    _, y_test_proper, y_pred_proper = proper_results
    
    mape_leak = mape(y_test_leak, y_pred_leak)
    mape_proper = mape(y_test_proper, y_pred_proper)
    
    logger.info(f"MAPE with leakage: {mape_leak:.2f}%")
    logger.info(f"MAPE without leakage: {mape_proper:.2f}%")
    logger.info(f"Difference in MAPE: {mape_proper - mape_leak:.2f}%")

# MAPE with data leakage: 16.67%
# MAPE without data leakage: 22.74%
# Difference in MAPE: 6.07%


# Function to create features (wrong way - with lookahead bias)
def create_features_with_lookahead(df):
    df['next_day_price'] = df['value'].shift(-1)  # Target (tomorrow's price)
    df['future_5day_ma'] = df['value'].rolling(window=5, center=True).mean()
    df['future_volatility'] = df['value'].rolling(window=10, center=True).std()
    return df

# Function to create features (correct way - no lookahead)
def create_features_proper(df):
    df['next_day_price'] = df['value'].shift(-1)  # Target (tomorrow's price)
    df['past_5day_ma'] = df['value'].rolling(window=5).mean()
    df['past_volatility'] = df['value'].rolling(window=10).std()
    return df

# Function to train and evaluate model with proper time series split
def evaluate_model(data, features, title, ax):
    # Remove NaN values
    data = data.dropna()
    
    # Proper time series split (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Prepare features and target
    X_train = train_data[features]
    y_train = train_data['next_day_price']
    X_test = test_data[features]
    y_test = test_data['next_day_price']
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    mape_score = mape(y_test, y_pred)
    
    # Plot results
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            'r--', label='Perfect Prediction')
    ax.set_title(f'{title}\nMAPE: {mape_score:.2f}%')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.legend()
    
    return mape_score, test_data.index, y_test, y_pred

if __name__ == "__main__":
    # Fetch US Natural Gas data
    gas_data = fetch_fred_data('PNGASUSUSDM')
    gas_data = gas_data.set_index('date')
    
    # Create both datasets
    data_with_lookahead = create_features_with_lookahead(gas_data.copy())
    data_proper = create_features_proper(gas_data.copy())

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Evaluate both models
    mape_lookahead, test_dates_look, y_test_look, y_pred_look = evaluate_model(
        data_with_lookahead, 
        ['future_5day_ma', 'future_volatility'],
        'Model with Lookahead Bias',
        ax1
    )

    mape_proper, test_dates_prop, y_test_prop, y_pred_prop = evaluate_model(
        data_proper, 
        ['past_5day_ma', 'past_volatility'],
        'Model without Lookahead Bias',
        ax2
    )

    plt.tight_layout()
    plt.show()

    # Print comparison
    logger.info(f"MAPE with lookahead bias: {mape_lookahead:.2f}%")
    logger.info(f"MAPE without lookahead bias: {mape_proper:.2f}%")
    logger.info(f"Difference in MAPE: {mape_proper - mape_lookahead:.2f}%")

    # Plot time series predictions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot actual vs predicted prices over time
    ax1.plot(test_dates_look, y_test_look, label='Actual Price', alpha=0.7)
    ax1.plot(test_dates_look, y_pred_look, label='Predicted (with lookahead)', linestyle='--')
    ax1.set_title(f'Price Predictions with Lookahead Bias (MAPE: {mape_lookahead:.2f}%)')
    ax1.legend()

    ax2.plot(test_dates_prop, y_test_prop, label='Actual Price', alpha=0.7)
    ax2.plot(test_dates_prop, y_pred_prop, label='Predicted (proper)', linestyle='--')
    ax2.set_title(f'Price Predictions without Lookahead Bias (MAPE: {mape_proper:.2f}%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Visualize the features
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot moving averages
    ax1.plot(gas_data.index, gas_data['value'], label='Price', alpha=0.5)
    ax1.plot(data_with_lookahead.index, data_with_lookahead['future_5day_ma'], 
             label='5-day MA (with lookahead)', linewidth=2)
    ax1.plot(data_proper.index, data_proper['past_5day_ma'], 
             label='5-day MA (proper)', linewidth=2)
    ax1.set_title('Price and Moving Averages')
    ax1.legend()

    # Plot volatility
    ax2.plot(data_with_lookahead.index, data_with_lookahead['future_volatility'], 
             label='Volatility (with lookahead)', linewidth=2)
    ax2.plot(data_proper.index, data_proper['past_volatility'], 
             label='Volatility (proper)', linewidth=2)
    ax2.set_title('Volatility Measures')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Performance Metrics:
# --------------------------------------------------
# MAPE with lookahead bias: 17.81%
# MAPE without lookahead bias: 36.03%
# Difference in MAPE: 18.22%

def granger_causality(data, max_lag=12):
    results = {}
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                test_result = grangercausalitytests(data[[col1, col2]], maxlag=max_lag, verbose=False)
                min_p_value = min([test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)])
                results[f"{col1} -> {col2}"] = min_p_value
    return results

def plot_correlations_and_scatter(data):
    # Create correlation matrix
    corr = data.corr()
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Correlation heatmap
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title("Correlation Heatmap")
    
    # Scatter plots
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax2.scatter(data['Japan Gas'], data['EM Gas'])
    ax2.set_xlabel('Japan Gas')
    ax2.set_ylabel('EM Gas')
    ax2.set_title('Japan Gas vs EM Gas')
    
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.scatter(data['Japan Gas'], data['US Loan Rate'])
    ax3.set_xlabel('Japan Gas')
    ax3.set_ylabel('US Loan Rate')
    ax3.set_title('Japan Gas vs US Loan Rate')
    
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.scatter(data['EM Gas'], data['US Loan Rate'])
    ax4.set_xlabel('EM Gas')
    ax4.set_ylabel('US Loan Rate')
    ax4.set_title('EM Gas vs US Loan Rate')
    
    plt.tight_layout()
    plt.show()

def plot_time_series(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot gas prices
    ax1.plot(data.index, data['Japan Gas'], label='Japan Gas')
    ax1.plot(data.index, data['EM Gas'], label='EM Gas')
    ax1.set_title('Natural Gas Prices Over Time')
    ax1.legend()
        # Plot US Loan Rate
#     ax2.plot(data.index, data['US Loan Rate'], label='US Loan Rate', color='green')
#     ax2.set_title('US Loan Rate Over Time')
#     ax2.legend()
#         plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Fetch data
    japan_gas = fetch_fred_data('PNGASJPUSDM')
    em_gas = fetch_fred_data('PNGASEUUSDM')
    us_loan_rate = fetch_fred_data('TERMCBPER24NS')
    
    # Combine data
    data = pd.concat([japan_gas, em_gas, us_loan_rate], axis=1)
    data.columns = ['Japan Gas', 'EM Gas', 'US Loan Rate']
    data = data.dropna()
    
    # Plot time series
    plot_time_series(data)
    
    # Plot correlations and scatter plots
    plot_correlations_and_scatter(data)
    
    # Perform and display Granger Causality tests
    gc_results = granger_causality(data)
    
    logger.info("\nGranger Causality Results (p-values):")
    logger.info("----------------------------------------")
    for pair, p_value in gc_results.items():
        significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
        logger.info(f"{pair}: {p_value:.4f} {significance}")
    logger.info("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1")

# Granger Causality Results (p-values):
# ----------------------------------------
# Japan Gas -> EM Gas: 0.0003 ***
# Japan Gas -> US Loan Rate: 0.0014 ***
# EM Gas -> Japan Gas: 0.0000 ***
# EM Gas -> US Loan Rate: 0.0008 ***
# US Loan Rate -> Japan Gas: 0.0081 ***
# US Loan Rate -> EM Gas: 0.0005 ***

# Significance levels: *** p<0.01, ** p<0.05, * p<0.1


warnings.filterwarnings('ignore')

def fetch_fred_data(series_id, start_date='2000-01-01', end_date=None):
    """Fetch FRED data using pandas_datareader."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
    df = df.rename(columns={series_id: 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()
    df = df.reset_index()  
    df = df.rename(columns={'DATE': 'date'})  
    return df

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_features(df, leakage=False):
    df = df.copy()
    if leakage:
        df['rolling_mean'] = df['value'].rolling(window=7, center=True).mean()
        df['volatility'] = df['value'].rolling(window=10, center=True).std()
    else:
        df['rolling_mean'] = df['value'].rolling(window=7).mean().shift(1)
        df['volatility'] = df['value'].rolling(window=10).std().shift(1)
    df['price_lag'] = df['value'].shift(1)
    df['monthly_return'] = df['value'].pct_change(periods=30)
    return df

def create_features_with_lookahead(df):
    df = df.copy()
    df['next_day_price'] = df['value'].shift(-1)
    df['future_5day_ma'] = df['value'].rolling(window=5, center=True).mean()
    df['future_volatility'] = df['value'].rolling(window=10, center=True).std()
    return df

def create_features_proper(df):
    df = df.copy()
    df['next_day_price'] = df['value'].shift(-1)
    df['past_5day_ma'] = df['value'].rolling(window=5).mean()
    df['past_volatility'] = df['value'].rolling(window=10).std()
    return df

def train_model(data, features, target='value'):
    data = data.dropna()
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return test_data.index, y_test, y_pred

def evaluate_model(data, features, title, ax):
    data = data.dropna()
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    X_train = train_data[features]
    y_train = train_data['next_day_price']
    X_test = test_data[features]
    y_test = test_data['next_day_price']
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape_score = mape(y_test, y_pred)
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    ax.set_title(f'{title}\nMAPE: {mape_score:.2f}%')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.legend()
    return mape_score, test_data.index, y_test, y_pred

def plot_features(data, leakage_data, proper_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(data.index, data['value'], label='Original Price', alpha=0.5)
    ax1.plot(leakage_data.index, leakage_data['rolling_mean'], label='Rolling Mean (leakage)', linewidth=2)
    ax1.plot(proper_data.index, proper_data['rolling_mean'], label='Rolling Mean (proper)', linewidth=2)
    ax1.set_title('Rolling Means Comparison')
    ax1.legend()
    ax2.plot(leakage_data.index, leakage_data['volatility'], label='Volatility (leakage)', linewidth=2)
    ax2.plot(proper_data.index, proper_data['volatility'], label='Volatility (proper)', linewidth=2)
    ax2.set_title('Volatility Comparison')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_predictions(leakage_results, proper_results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    dates_leak, y_test_leak, y_pred_leak = leakage_results
    dates_proper, y_test_proper, y_pred_proper = proper_results
    mape_leak = mape(y_test_leak, y_pred_leak)
    mape_proper = mape(y_test_proper, y_pred_proper)
    ax1.plot(dates_leak, y_test_leak, label='Actual', alpha=0.7)
    ax1.plot(dates_leak, y_pred_leak, '--', label=f'Leakage (MAPE: {mape_leak:.2f}%)')
    ax1.plot(dates_proper, y_pred_proper, '--', label=f'Proper (MAPE: {mape_proper:.2f}%)')
    ax1.legend()
    ax2.scatter(y_test_leak, y_pred_leak, alpha=0.5, label='Leakage')
    ax2.scatter(y_test_proper, y_pred_proper, alpha=0.5, label='Proper')
    ax2.plot([min(y_test_leak.min(), y_test_proper.min()), max(y_test_leak.max(), y_test_proper.max())],
             [min(y_test_leak.min(), y_test_proper.min()), max(y_test_leak.max(), y_test_proper.max())], 'r--')
    ax2.set_title('Prediction Comparison')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def granger_causality(data, max_lag=12):
    results = {}
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                test_result = grangercausalitytests(data[[col1, col2]], maxlag=max_lag, verbose=False)
                min_p = min([test_result[i + 1][0]['ssr_ftest'][1] for i in range(max_lag)])
                results[f"{col1} -> {col2}"] = min_p
    return results

def plot_time_series(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.plot(data.index, data['Japan Gas'], label='Japan Gas')
    ax1.plot(data.index, data['EM Gas'], label='EM Gas')
    ax1.legend()
    ax1.set_title('Gas Prices')
    ax2.plot(data.index, data['US Loan Rate'], label='US Loan Rate', color='green')
    ax2.legend()
    ax2.set_title('US Loan Rate')
    plt.tight_layout()
    plt.show()

def plot_correlations_and_scatter(data):
    corr = data.corr()
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title("Correlation Heatmap")
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax2.scatter(data['Japan Gas'], data['EM Gas'])
    ax2.set_title('Japan Gas vs EM Gas')
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.scatter(data['Japan Gas'], data['US Loan Rate'])
    ax3.set_title('Japan Gas vs US Loan Rate')
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.scatter(data['EM Gas'], data['US Loan Rate'])
    ax4.set_title('EM Gas vs US Loan Rate')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Leakage analysis ---
    japan_gas = fetch_fred_data('PNGASJPUSDM')
    data_with_leakage = create_features(japan_gas, leakage=True)
    data_proper = create_features(japan_gas, leakage=False)
    plot_features(japan_gas, data_with_leakage, data_proper)
    features = ['rolling_mean', 'volatility', 'price_lag', 'monthly_return']
    leakage_results = train_model(data_with_leakage, features)
    proper_results = train_model(data_proper, features)
    plot_predictions(leakage_results, proper_results)

    # --- Lookahead bias analysis ---
    gas_data = fetch_fred_data('PNGASUSUSDM').set_index('date')
    data_lookahead = create_features_with_lookahead(gas_data)
    data_correct = create_features_proper(gas_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    m1, d1, y1, p1 = evaluate_model(data_lookahead, ['future_5day_ma', 'future_volatility'], 'With Lookahead', ax1)
    m2, d2, y2, p2 = evaluate_model(data_correct, ['past_5day_ma', 'past_volatility'], 'Without Lookahead', ax2)
    plt.tight_layout()
    plt.show()

    # --- Granger causality ---
    em_gas = fetch_fred_data('PNGASEUUSDM')
    us_rate = fetch_fred_data('TERMCBPER24NS')
    combined = pd.concat([japan_gas.set_index('date')['value'], 
                          em_gas.set_index('date')['value'], 
                          us_rate.set_index('date')['value']], axis=1)
    combined.columns = ['Japan Gas', 'EM Gas', 'US Loan Rate']
    combined = combined.dropna()
    plot_time_series(combined)
    plot_correlations_and_scatter(combined)
    results = granger_causality(combined)
    logger.info("\nGranger Causality (p-values):")
    for k, v in results.items():
        flag = "***" if v < 0.01 else "**" if v < 0.05 else "*" if v < 0.1 else ""
        logger.info(f"{k}: {v:.4f} {flag}")
