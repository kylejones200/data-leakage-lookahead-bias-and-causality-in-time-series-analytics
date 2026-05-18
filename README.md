# Data Leakage, Lookahead Bias, and Causality

This project demonstrates data leakage detection and prevention techniques.

## Business context

Time series analytics presents unique challenges and pitfalls that can compromise the validity of insights and predictions. Among these, data leakage, lookahead bias, and causality violations are particularly common and detrimental.

Data leakage occurs when information that would not be available at the time of prediction inadvertently influences the model. In time series analysis, this issue is especially insidious because of the temporal nature of data. Leakage can make a model appear far more accurate during training than it actually is when deployed in real-world scenarios.

- Including future values: Using future data points as predictors for past or present events. - Feature engineering pitfalls: Accidentally deriving features that implicitly incorporate future information (e.g., calculating a rolling average using future timestamps). - Improper splitting: Shuffling data without considering its sequential nature, leading to overlap between training and testing datasets.

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
- Lookahead Bias: Using future information
- Target Leakage: Including target in features
- Improper Scaling: Scaling before train/test split
- Data Snooping: Using full dataset for feature engineering

## Caveats

- By default, generates synthetic time series data.
- Always split data before feature engineering.
- Validate models on holdout data.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).