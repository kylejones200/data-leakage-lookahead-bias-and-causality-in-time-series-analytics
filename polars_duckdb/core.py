"""Data leakage and lookahead bias using Polars and DuckDB.

The core insight: DuckDB window frames make causality structurally explicit.

  Causal (no leakage):
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
      → window physically cannot see the current or future rows

  Lookahead (leakage):
      ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING  (or LEAD())
      → window explicitly reaches into the future

sklearn.linear_model and sklearn.metrics are replaced by
numpy lstsq + DuckDB REGR_R2 / AVG(POWER(...)) / AVG(ABS(...)).
"""

import duckdb
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple


def create_features(df: pl.DataFrame, date_col: str, value_col: str,
                    leakage: bool = False) -> pl.DataFrame:
    """
    Build predictive features.

    leakage=False  →  all windows end at 1 PRECEDING; structurally causal.
    leakage=True   →  rolling stats use center=True equivalent (lookahead).
    """
    if not leakage:
        result = duckdb.sql(f"""
            SELECT
                "{date_col}",
                "{value_col}",
                AVG("{value_col}")
                    OVER (ORDER BY "{date_col}" ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
                    AS rolling_mean,
                STDDEV_SAMP("{value_col}")
                    OVER (ORDER BY "{date_col}" ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
                    AS volatility,
                LAG("{value_col}", 1) OVER (ORDER BY "{date_col}")
                    AS price_lag,
                ("{value_col}" - LAG("{value_col}", 30) OVER (ORDER BY "{date_col}"))
                    / NULLIF(LAG("{value_col}", 30) OVER (ORDER BY "{date_col}"), 0)
                    AS monthly_return
            FROM df
            ORDER BY "{date_col}"
        """).pl().drop_nulls()
    else:
        # Lookahead: centered rolling mean leaks future data
        result = duckdb.sql(f"""
            SELECT
                "{date_col}",
                "{value_col}",
                AVG("{value_col}")
                    OVER (ORDER BY "{date_col}" ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)
                    AS rolling_mean,
                STDDEV_SAMP("{value_col}")
                    OVER (ORDER BY "{date_col}" ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING)
                    AS volatility,
                LAG("{value_col}", 1) OVER (ORDER BY "{date_col}")
                    AS price_lag,
                ("{value_col}" - LAG("{value_col}", 30) OVER (ORDER BY "{date_col}"))
                    / NULLIF(LAG("{value_col}", 30) OVER (ORDER BY "{date_col}"), 0)
                    AS monthly_return
            FROM df
            ORDER BY "{date_col}"
        """).pl().drop_nulls()
    return result


def create_features_with_lookahead(
    df: pl.DataFrame, date_col: str, value_col: str
) -> pl.DataFrame:
    """Explicit future-leaking features: LEAD() and forward window."""
    return duckdb.sql(f"""
        SELECT
            "{date_col}",
            "{value_col}",
            LEAD("{value_col}", 1) OVER (ORDER BY "{date_col}")
                AS next_day_price,
            AVG("{value_col}")
                OVER (ORDER BY "{date_col}" ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)
                AS future_rolling_mean
        FROM df
        ORDER BY "{date_col}"
    """).pl().drop_nulls()


def train_model(
    df: pl.DataFrame, feature_cols: list, target_col: str
) -> Tuple[np.ndarray, float, Dict]:
    """numpy lstsq for coefficients; DuckDB for all evaluation metrics."""
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()
    X_aug = np.column_stack([X, np.ones(len(X))])
    result, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    coefs, intercept = result[:-1], float(result[-1])

    # score via DuckDB
    pred_expr = " + ".join(
        f"{c} * \"{col}\"" for c, col in zip(coefs, feature_cols)
    ) + f" + {intercept}"

    metrics = duckdb.sql(f"""
        WITH preds AS (
            SELECT "{target_col}" AS actual, {pred_expr} AS predicted FROM df
        )
        SELECT
            REGR_R2(actual, predicted)              AS r2,
            SQRT(AVG(POWER(actual - predicted, 2))) AS rmse,
            AVG(ABS(actual - predicted))            AS mae
        FROM preds
    """).pl().row(0, named=True)

    return coefs, intercept, metrics


def plot_leakage_comparison(
    metrics_no_leakage: Dict,
    metrics_with_leakage: Dict,
    title: str,
    output_path: Path,
    plot: bool = False,
):
    if not plot:
        return
    categories = ["R²", "RMSE", "MAE"]
    no_leak   = [metrics_no_leakage["r2"],   metrics_no_leakage["rmse"],   metrics_no_leakage["mae"]]
    with_leak = [metrics_with_leakage["r2"], metrics_with_leakage["rmse"], metrics_with_leakage["mae"]]
    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, no_leak,   w, label="No Leakage",   color="#4A90A4", alpha=0.7)
    ax.bar(x + w / 2, with_leak, w, label="With Leakage", color="#D4A574", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
