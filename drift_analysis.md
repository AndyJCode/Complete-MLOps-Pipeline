# Data Drift Analysis

# Overview
This analysis evaluates data drift across three monthly batches (`month1`, `month2`, `month3`) compared to the training dataset in the heart disease prediction pipeline.

Drift detection was performed using statistical tests (KS test), with results summarized as:
- **Feature Drift Share**: proportion of features showing drift
- **Dataset Drift Detection**: overall drift flag

---

# Summary of Results

| Batch   | Rows | Feature Drift Share | Dataset Drift |
|--------|------|--------------------|--------------|
| Month 1 | 61   | 0.143              | No           |
| Month 2 | 61   | 0.143              | No           |
| Month 3 | 62   | 0.143              | No           |

- Approximately **14.3% of features (~2 out of 14)** show drift in each batch
- **No full dataset drift detected** in any batch

---

# Key Observations

1) Cholesterol (`chol`)
- Observed values include **0**, which likely represent missing or incorrectly encoded data
- This introduces:
  - Artificial distribution shifts
  - Noise in statistical tests
- Likely contributor to detected feature drift

---

2) Resting Blood Pressure (`trestbps`)
- Month 3 shows unusually high values:
  - Example: `260`, `205`
- These exceed normal physiological ranges and indicate:
  - Potential outliers
  - Data entry or measurement inconsistencies

---

3) Age (`age`)
- Extreme values observed:
  - Example: `107`
- While rare, such values may:
  - Skew distribution statistics
  - Contribute to minor drift signals

---

4) Statistical Warnings
- Runtime warnings observed:

5) Improve Data Quality Fixes
- Improved or handled outliars more approtriately based on use cases. 
- Instead of median I could have considered mean as well, or removed them entirely.

6) Conclusion 
- No significant data drift detected from all 3 months
- Only small drift due to data quality issues
- Focus should have been on data cleaning and validation before other implementations.