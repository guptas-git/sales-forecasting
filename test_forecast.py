"""Sanity checks on synthetic data with a known answer. Run: python test_forecast.py"""
import numpy as np
import pandas as pd

import forecast as f

MONTHS = pd.date_range("2013-01-01", periods=60, freq="MS")
SEASON = np.array([0.8, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.15, 1.05, 1.0, 0.95, 0.9])


def synthetic(trend, noise=0.0):
    t = np.arange(len(MONTHS))
    base = (1000 + trend * t) * SEASON[MONTHS.month - 1]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {item: base * item * (1 + noise * rng.standard_normal(len(t))) for item in (1, 2, 3)}, index=MONTHS
    ).asfreq("MS")


def test_seasonal_naive_is_exact_on_pure_seasonality():
    bt = f.backtest(synthetic(trend=0), {"seasonal_naive": f.seasonal_naive})
    assert bt.mape.max() < 1e-9


def test_models_track_trend_and_seasonality():
    y = synthetic(trend=15, noise=0.01)
    for name, model in f.MODELS.items():
        pred = model(y.iloc[:-f.H])
        assert pred.shape == (f.H, 3) and not pred.isna().any().any(), name
        assert list(pred.index) == list(y.index[-f.H:]), name
    bt = f.backtest(y).groupby("model").mape.mean()
    # trees can't extrapolate a steep trend, so xgboost only has to beat the baseline here
    assert bt["sarima"] < 3 and bt["xgboost"] < bt["seasonal_naive"], bt


if __name__ == "__main__":
    test_seasonal_naive_is_exact_on_pure_seasonality()
    test_models_track_trend_and_seasonality()
    print("ok")
