#!/usr/bin/env python3
"""Data leakage and lookahead bias — Polars + DuckDB rewrite (no sklearn)."""

import sys
import argparse
import yaml
import logging
import numpy as np
import polars as pl
from datetime import date, timedelta
from pathlib import Path

from core import create_features, create_features_with_lookahead, train_model, plot_leakage_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
FEATURE_COLS = ["rolling_mean", "volatility", "price_lag", "monthly_return"]


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Data leakage — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    value_col = config["data"]["value_column"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["figures_dir"])
    output_dir.mkdir(exist_ok=True)

    if args.data_path and args.data_path.exists():
        df = pl.read_csv(args.data_path, try_parse_dates=True)
    elif config["data"]["generate_synthetic"]:
        rng = np.random.default_rng(config["data"]["seed"])
        n = config["data"]["n_periods"]
        start = date(2023, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n)]
        values = 100 + np.cumsum(rng.normal(0, 2, n))
        df = pl.DataFrame({"date": dates, value_col: values.tolist()})
    else:
        raise ValueError("No data source specified")

    # ── causal features (no leakage) ─────────────────────────────────────────
    df_clean = create_features(df, "date", value_col, leakage=False)
    _, _, metrics_clean = train_model(df_clean, FEATURE_COLS, value_col)
    logging.info("Model WITHOUT leakage (causal windows: END AT 1 PRECEDING):")
    logging.info(f"  R²:   {metrics_clean['r2']:.4f}")
    logging.info(f"  RMSE: {metrics_clean['rmse']:.4f}")
    logging.info(f"  MAE:  {metrics_clean['mae']:.4f}")

    # ── lookahead features (leakage) ─────────────────────────────────────────
    if config["analysis"]["compare_leakage"]:
        df_leak = create_features(df, "date", value_col, leakage=True)
        _, _, metrics_leak = train_model(df_leak, FEATURE_COLS, value_col)
        logging.info("\nModel WITH leakage (centered windows: INCLUDE FOLLOWING rows):")
        logging.info(f"  R²:   {metrics_leak['r2']:.4f}  ← inflated by future data")
        logging.info(f"  RMSE: {metrics_leak['rmse']:.4f}")
        logging.info(f"  MAE:  {metrics_leak['mae']:.4f}")

        r2_inflation = metrics_leak["r2"] - metrics_clean["r2"]
        logging.info(f"\nR² inflation from leakage: +{r2_inflation:.4f}")

        plot_leakage_comparison(
            metrics_clean, metrics_leak,
            "Model Metrics: Causal vs Lookahead Features",
            output_dir / "leakage_comparison.png",
        )

    logging.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
