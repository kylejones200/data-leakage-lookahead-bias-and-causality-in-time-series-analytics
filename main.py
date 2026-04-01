#!/usr/bin/env python3
"""
Data Leakage, Lookahead Bias, and Causality Analysis

Main entry point for running data leakage and causality analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Data Leakage and Causality Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        values = 100 + np.cumsum(np.random.normal(0, 2, config['data']['n_periods']))
        df = pd.DataFrame({
            'date': dates,
            config['data']['value_column']: values
        })
    else:
        raise ValueError("No data source specified")
    
        df_no_leakage = create_features(df, leakage=False).dropna()
    X_no_leak = df_no_leakage[['rolling_mean', 'volatility', 'price_lag', 'monthly_return']].values
    y_no_leak = df_no_leakage[config['data']['value_column']].values
    
    model_no_leak, metrics_no_leak = train_model(X_no_leak, y_no_leak)
    logging.info(f"\nModel WITHOUT Leakage:")
    logging.info(f"  R²: {metrics_no_leak['r2']:.4f}")
    logging.info(f"  RMSE: {metrics_no_leak['rmse']:.4f}")
    
    if config['analysis']['compare_leakage']:
                df_with_leakage = create_features_with_lookahead(df).dropna()
        if 'future_rolling_mean' in df_with_leakage.columns:
            X_with_leak = df_with_leakage[['future_rolling_mean']].values
            y_with_leak = df_with_leakage[config['data']['value_column']].values
            
            model_with_leak, metrics_with_leak = train_model(X_with_leak, y_with_leak)
            logging.info(f"\nModel WITH Leakage:")
            logging.info(f"  R²: {metrics_with_leak['r2']:.4f}")
            logging.info(f"  RMSE: {metrics_with_leak['rmse']:.4f}")
            
                        plot_leakage_comparison(metrics_no_leak, metrics_with_leak,
                                  "Data Leakage Comparison", output_dir / 'leakage_comparison.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

