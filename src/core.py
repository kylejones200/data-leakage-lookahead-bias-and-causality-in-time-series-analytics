"""Core functions for data leakage, lookahead bias, and causality analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_features(df: pd.DataFrame, leakage: bool = False) -> pd.DataFrame:
    """Create features with or without data leakage."""
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

def create_features_with_lookahead(df: pd.DataFrame) -> pd.DataFrame:
    """Create features improperly with lookahead bias."""
    df = df.copy()
    df['next_day_price'] = df['value'].shift(-1)
    df['future_rolling_mean'] = df['value'].rolling(window=7, center=True).mean()
    return df

def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, Dict]:
    """Train model and return metrics."""
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
        'mae': np.mean(np.abs(y - y_pred))
    }
    
    return model, metrics

def plot_leakage_comparison(metrics_no_leakage: Dict, metrics_with_leakage: Dict,
                           title: str, output_path: Path):
 """Plot comparison of models with and without leakage """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['R²', 'RMSE', 'MAE']
    no_leakage = [metrics_no_leakage['r2'], metrics_no_leakage['rmse'], metrics_no_leakage['mae']]
    with_leakage = [metrics_with_leakage['r2'], metrics_with_leakage['rmse'], metrics_with_leakage['mae']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, no_leakage, width, label='No Leakage', color="#4A90A4", alpha=0.7)
    ax.bar(x + width/2, with_leakage, width, label='With Leakage', color="#D4A574", alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

