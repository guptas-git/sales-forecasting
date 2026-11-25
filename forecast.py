"""Monthly demand forecasting per item: backtest 4 models, then forecast the next 6 months with the best one."""
import logging
import warnings
from pathlib import Path

# Silence these before importing prophet: it logs a plotly error on import (its plots aren't used here),
# and cmdstanpy resets its own log level on the first fit.
logging.getLogger("prophet.plot").disabled = True
logging.getLogger("cmdstanpy").disabled = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")  # after imports: statsmodels re-enables its own warnings on import

H = 6      # forecast horizon, months
FOLDS = 3  # rolling origins, each H months apart
ROOT = Path(__file__).parent


def load(path):
    """Daily store-item sales -> monthly sales, one column per item (summed over stores)."""
    df = pd.read_csv(path, parse_dates=["date"])
    monthly = df.groupby(["item", pd.Grouper(key="date", freq="MS")])["sales"].sum()
    return monthly.unstack("item").asfreq("MS")


def future_index(train):
    return pd.date_range(train.index[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")


# Every model takes history up to the origin (month x item) and returns the next H months (month x item).

def seasonal_naive(train):
    """Baseline: same month last year. Every model has to beat this."""
    n = len(train)
    return pd.DataFrame(train.iloc[n - 12:n - 12 + H].values, index=future_index(train), columns=train.columns)


def sarima(train):
    """Airline model (0,1,1)(0,1,1)12 on log sales, one per item."""
    out = {}
    for item, y in train.items():
        fit = SARIMAX(np.log(y), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        out[item] = np.exp(fit.forecast(H).values)
    return pd.DataFrame(out, index=future_index(train))


def prophet(train):
    out = {}
    for item, y in train.items():
        m = Prophet(seasonality_mode="multiplicative", weekly_seasonality=False, daily_seasonality=False)
        m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        out[item] = m.predict(pd.DataFrame({"ds": future_index(train)}))["yhat"].values
    return pd.DataFrame(out, index=future_index(train))


def xgboost(train):
    """One model pooled across all items.

    Features only use values at least H months old, so all H future months are predicted
    directly (no recursive forecasting). Target and lags are divided by the item's trailing
    12-month level, which makes items of different size comparable for a single model.
    """
    full = train.reindex(train.index.append(future_index(train)))  # future rows are NaN
    long = full.rename_axis("date").melt(ignore_index=False, var_name="item", value_name="y").reset_index()
    g = long.groupby("item")["y"]
    long["level"] = g.transform(lambda s: s.shift(H).rolling(12).mean())
    long["lag_h"] = g.shift(H) / long.level
    long["lag_12"] = g.shift(12) / long.level
    long["month"] = long.date.dt.month
    cols = ["lag_h", "lag_12", "month"]

    fit = long[long.y.notna() & long.level.notna()]
    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(fit[cols], fit.y / fit.level)

    pred = long[long.date > train.index[-1]].copy()
    pred["yhat"] = model.predict(pred[cols]) * pred.level
    return pred.pivot(index="date", columns="item", values="yhat").rename_axis(index=None)


MODELS = {"seasonal_naive": seasonal_naive, "sarima": sarima, "prophet": prophet, "xgboost": xgboost}


def mape(actual, pred):
    # shortcut: MAPE assumes positive monthly sales; use WAPE for data containing zeros.
    return (abs(actual - pred) / actual).mean() * 100


def backtest(y, models=MODELS):
    """Rolling-origin backtest: train on everything before each origin, score the next H months."""
    rows = []
    for k in range(FOLDS, 0, -1):
        origin = len(y) - k * H
        train, test = y.iloc[:origin], y.iloc[origin:origin + H]
        for name, model in models.items():
            for item, err in mape(test, model(train)).items():
                rows.append({"fold": FOLDS - k + 1, "test_start": test.index[0].date(), "model": name, "item": item, "mape": err})
    return pd.DataFrame(rows)


def plot(y, forecast, summary, best, path):
    from matplotlib.ticker import StrMethodFormatter

    labels = {"seasonal_naive": "Same month last year", "sarima": "SARIMA", "prophet": "Prophet", "xgboost": "XGBoost"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    a.plot(y.index, y.sum(axis=1), color="#315e78", label="Observed sales")
    a.plot(forecast.index, forecast.sum(axis=1), "--", color="#c48b4a", label=f"{labels[best]} forecast")
    a.axvspan(forecast.index[0], forecast.index[-1], color="#c48b4a", alpha=0.08)
    a.set_title(f"Monthly demand across {y.shape[1]} items")
    a.set_ylabel("Units sold")
    a.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    a.legend()
    errors = summary["mean"].sort_values(ascending=False)
    errors.rename(index=labels).plot.barh(ax=b, color=["#315e78" if name == best else "#aebbc3" for name in errors.index])
    b.bar_label(b.containers[0], labels=[f"{value:.2f}%" for value in errors], padding=4)
    b.set_xlim(0, errors.max() * 1.25)
    b.set_title("Three backtest windows, six months each")
    b.set_xlabel("Mean MAPE (%) · lower is better")
    b.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    y = load(ROOT / "data" / "train.csv")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    print(f"{y.shape[1]} items, {len(y)} months ({y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m})")

    bt = backtest(y)
    bt.to_csv(out / "backtest.csv", index=False)
    summary = bt.groupby("model").mape.agg(["mean", "median", "max"]).sort_values("mean")
    summary.round(2).to_csv(out / "summary.csv")
    print(summary.round(2))

    best = summary.index[0]
    per_item = bt.groupby(["item", "model"]).mape.mean().unstack()
    wins = (per_item[best] < per_item["seasonal_naive"]).sum()
    print(f"\nBest: {best}. It beats the seasonal-naive baseline on {wins}/{len(per_item)} items.")

    forecast = MODELS[best](y)
    forecast.round().astype(int).to_csv(out / f"forecast_next_{H}_months.csv")
    plot(y, forecast, summary, best, out / "forecast.png")


if __name__ == "__main__":
    main()
