# Viva preparation

Owner: Satya Yadav. The viva is **individually assessed**, so every member must be able to
explain any part of the pipeline — including a teammate's notebook.

Plan at least **two mock vivas** where each person explains another member's part and
answers questions on it.

## Dataset and EDA

- What is the target, and why that definition rather than another standard?
- Where could leakage have entered, and what did we do about it?
- How were detection-limit values (`<0.5`) handled, and why that rule?
- Why flag impossible values instead of dropping those rows?
- Why is imputation in the pipeline rather than in the cleaning step?

## Split and preprocessing

- Why a site/time-separated split instead of a random one?
- What breaks if the scaler is fitted before splitting?
- How do we know every member trained on identical data?
- How many times has the test set been used, and why does that matter?

## Models

- Random Forest: how does bagging reduce variance? What is out-of-bag error?
- Gradient boosting: what loss is being minimised? What do shrinkage and regularization do?
- Why does one model beat the other on this dataset?
- What does the naive baseline tell us, and did every model beat it?
- Which hyperparameters mattered most, and on which split were they chosen?

## Metrics and interpretation

- What does each metric measure, and why does it fit this target?
- Why not accuracy (if classification)? Why RMSE *and* MAE (if regression)?
- What does a SHAP value actually represent? How does it differ from tree feature importance?
- Where does the model fail — which sites, which time periods, which value ranges?

## Theory notes (for the report, one per member)

| Note | Owner |
| :--- | :--- |
| Split design and leakage | Parthraj Singh Bhati |
| SHAP and feature importance | Parthraj Singh Bhati |
| Missing data handling | Shaik Tajuddin |
| Random Forest: bagging, variance reduction, OOB error | Himanshu Pandey |
| Gradient boosting: loss, additive trees, shrinkage, regularization | Jivit Rana |
| Metrics: what each measures and why it fits this target | Satya Yadav |
