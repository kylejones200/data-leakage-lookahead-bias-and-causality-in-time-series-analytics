# Data Leakage, Lookahead Bias, and Causality

This project demonstrates data leakage detection and prevention techniques.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Data leakage analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Feature engineering options
- Leakage comparison settings
- Output settings

## Data Leakage

Common leakage sources:
- **Lookahead Bias**: Using future information
- **Target Leakage**: Including target in features
- **Improper Scaling**: Scaling before train/test split
- **Data Snooping**: Using full dataset for feature engineering

## Caveats

- By default, generates synthetic time series data.
- Always split data before feature engineering.
- Validate models on holdout data.
