# Sales Forecasting & Demand Planning

Forecasts monthly demand for 50 products six months ahead. It compares SARIMA, Prophet and XGBoost against a naive baseline using rolling-origin backtesting.

> This rebuilds, on public data, the forecasting approach I used as a Data Analyst at Think and Wash. The original work used internal company data that I can't share.

![Forecast and backtest error](results/forecast.png)

## Problem

Capacity and staffing are planned six months out. The planners need a monthly demand forecast per product, plus evidence that it beats the rule they already use: "same as last year".

## Data

[Kaggle: Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data): 913,000 daily sales rows covering 10 stores and 50 items, 2013–2017. Sales are summed over stores into **60 months × 50 items**. The data is very regular, with little noise, so error levels are lower than real retail data would give.

## Approach

| Model | What it is |
|---|---|
| Seasonal naive | Same month last year. This is the baseline every model has to beat. |
| SARIMA | The "airline model" (0,1,1)(0,1,1)₁₂ on log sales, one model per item |
| Prophet | Multiplicative yearly seasonality, one model per item |
| XGBoost | **One model pooled across all 50 items.** Features are at least 6 months old, so all 6 months are predicted directly, with no recursion. Values are scaled by each item's trailing 12-month level, so large and small items share one model. Pooling works because every item has the same seasonal shape: the [EDA notebook](notebooks/eda.ipynb) shows a max spread of 0.028. |

**Backtesting:** there are three 6-month test windows (Jul–Dec 2016, Jan–Jun 2017, Jul–Dec 2017). Each model is trained only on months before its window, then scored with MAPE per item.

## Results

| Model | Mean MAPE | Median | Worst item |
|---|---|---|---|
| **SARIMA** | **2.71%** | 1.44% | 7.81% |
| XGBoost | 2.80% | 2.81% | 6.12% |
| Prophet | 2.99% | 2.43% | 6.39% |
| Seasonal naive | 4.89% | 3.91% | 9.21% |

- SARIMA beats the baseline on **50 of 50 items** and cuts average error by 45%.
- XGBoost is close behind, with the lowest worst-case error. It's also one model to maintain instead of 50.

**Where the models failed:** in Jan–Jun 2017, every model got worse and the naive baseline won.

| Window | SARIMA | XGBoost | Prophet | Naive |
|---|---|---|---|---|
| Jul–Dec 2016 | 1.34% | 2.85% | 2.25% | 7.73% |
| Jan–Jun 2017 | 5.72% | 3.85% | 4.92% | **3.15%** |
| Jul–Dec 2017 | 1.06% | 1.70% | 1.81% | 3.80% |

Year-on-year growth fell from 8.6% in 2016 to 3.6% in 2017. The models carried 2016's growth forward; SARIMA over-forecast every month of that window by 4–7%. A model trained on history can't anticipate a change in growth rate; production forecasts need error monitoring and retraining to catch it.

## How to run

Tested with Python 3.12. Put Kaggle's `train.csv` in `data/` first.

```bash
pip install -r requirements.txt
python forecast.py       # backtest + 6-month forecast → results/  (~1.5 min)
python test_forecast.py  # sanity checks on synthetic data with known answers
```

`results/` contains:
- `backtest.csv`: the error for every model, window and item
- `summary.csv`
- `forecast_next_6_months.csv`: Jan–Jun 2018, made with the best model
- `forecast.png`

## What I'd do next

- **Error monitoring:** track monthly forecast error and alert on drift. This would have flagged the 2017 slowdown within a month or two.
- **Ensemble:** average SARIMA and XGBoost, since they fail in different windows.
- **Prediction intervals:** size capacity buffers from a range instead of a single number.
- **Daily rates:** model sales per day instead of monthly totals, so leap-year Februaries don't skew the comparison.
