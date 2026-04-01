# Data Leakage, Lookahead Bias, and Causality in Time Series Analytics Smart ways to avoid common mistakes in forecasting

### Data Leakage, Lookahead Bias, and Causality in Time Series Analytics
#### Smart ways to avoid common mistakes in forecasting
Time series analytics presents unique challenges and pitfalls that can
compromise the validity of insights and predictions. Among these, data
leakage, lookahead bias, and causality violations are particularly
common and detrimental.

### Data Leakage: A Silent Saboteur
Data leakage occurs when information that would not be available at the
time of prediction inadvertently influences the model. In time series
analysis, this issue is especially insidious because of the temporal
nature of data. Leakage can make a model appear far more accurate during
training than it actually is when deployed in real-world scenarios.

**Examples of Data Leakage in Time Series:**

- **Including future values:** Using future data points as predictors
  for past or present events.
- **Feature engineering pitfalls:** Accidentally deriving features that
  implicitly incorporate future information (e.g., calculating a rolling
  average using future timestamps).
- **Improper splitting:** Shuffling data without considering its
  sequential nature, leading to overlap between training and testing
  datasets.

### Impact
Models affected by data leakage fail in production environments because
they rely on information they won't have access to when making real-time
predictions.

### Detection and Mitigation
- **Time-aware splitting:** Use methods like walk-forward validation or
  time-based splits to ensure chronological separation of training,
  validation, and test sets.
- **Rigorous feature engineering:** Ensure that features are derived
  only from past or present data relative to the prediction
  target.
- **Audit your pipeline:** Systematically review data flow to identify
  potential sources of leakage.

Let's look at an actual example. This uses the price of natural gas in
Japan. Data is from FRED (as of 2024--01--24).

In these examples, I'll show the WRONG way and the RIGHT way to do
things.




### Lookahead Bias --- Seeing the Future
Lookahead bias, a specific type of data leakage, occurs when the model
has access to future information during training or evaluation. This is
particularly problematic in forecasting applications, where the entire
premise is to predict unknown future values.

**Examples of Lookahead Bias:**

- **Improper labels:** Using the value of the target variable from the
  future when training the model.
- **Causal confusion:** Including predictors that are only generated
  after the event has occurred, such as stock prices or reactionary
  metrics.

### Impact
Lookahead bias undermines the credibility of models. A model that "knows
the future" during testing will exhibit inflated performance metrics,
leading to false confidence in its capabilities.

### Strategies to Prevent Lookahead Bias
- **Lagged features:** Ensure that all features represent information
  available at or before the time of prediction.
- **Careful backtesting:** Validate models using realistic scenarios
  where only historical data is accessible.
- **Feature audits:** Regularly check whether any feature is
  inadvertently derived from future values.


Lookahead is like cheating --- the model already knows the answer. So we
expect the MAPE to be better for look ahead.



And it is! The lookahead is 18.22 percentage points better than the
proper model.

### Causality: Ensuring Sound Relationships
In time series analytics, understanding causality is crucial for
creating reliable and actionable models. Predictive relationships
without causal underpinnings often fail when the underlying system
changes, as correlations may not hold over time.

**Challenges in Time Series Causality:**

- **Spurious correlations:** Random correlations between variables may
  appear significant in small samples.
- **Confounders:** Hidden variables may influence both the predictor
  and the target, creating misleading associations.
- **Reverse causality:** A predictor may be affected by the target
  variable rather than vice versa.

### Tools for Causality Analysis
- **Granger Causality:** A statistical test to determine whether one
  time series can predict another.
- **Directed Acyclic Graphs (DAGs):** Graphical models to identify
  potential causal relationships.
- **Counterfactual analysis:** Assess the impact of interventions or
  changes in predictor variables.

### Best Practices for Causality in Time Series
- **Domain expertise:** Collaborate with subject matter experts to
  validate causal assumptions.
- **Experimental design:** Use A/B testing or natural experiments to
  establish causality.
- **Robust modeling:** Incorporate techniques like structural equation
  modeling or Bayesian networks to account for confounding
  factors.

### Looking at a real example.
Let's look at the correlation between Natural Gas prices in Asia and
Europe and the US Finance Rate on Personal Loans at Commercial Banks. I
would expect a high correlation for the two natural gas indexes. There
is no reason that the price of LNG in Japan is dependent on the US
finance rate (well there could be some macroeconomic reason about
overall health of US economic indicators but practically, there isn't a
causal relationship).



The correlation plot shows the high correlation for Japan and EU gas
(the sign of an efficient market). As suspected, there is a lower
correlation between the gas price and the US loan rate.


And yet, with all this, we still get statistically significant Granger
Causality results. This indicates that there is low probability that the
observed relationship happened by change. Translation, there is evidence
to suggest a causal relationship between these variables.


### Practical Recommendations
1.  [**Adopt Time-Aware Validation:** Always split data based on time,
    keeping future data separate from the training set.]
2.  [**Audit Data Pipelines:** Regularly review each step of your
    pipeline to check for inadvertent leakage or bias.]
3.  [**Use Lagged Features:** Ensure all features represent information
    available at the time of prediction.]
4.  [**Validate with Experts:** Collaborate with domain experts to
    verify causal assumptions and identify potential
    confounders.]
5.  [**Monitor Models Post-Deployment:** Continuously evaluate model
    performance to detect issues arising from unseen biases or changing
    causal relationships.]

### Conclusion
Data leakage, lookahead bias, and causality violations make your models
appear better than they really are. Python will create results for
you --- even if those results are not following proper procedure. It is
up to you to implement best practices, use domain expertise, and
validate assumptions.
