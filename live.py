"""Live data: download WHO weekly COVID data, build weekly features, train, predict."""
import time
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

WHO_URL = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
CACHE = Path("data/who_weekly.csv")
MODEL = Path("models/weekly_model.joblib")
REQUIRED = ["Date_reported", "Country", "New_cases", "New_deaths", "Cumulative_cases"]
FEATS = ["lag_0", "lag_1", "lag_2", "lag_3", "death_lag_0", "growth", "log_cum", "week"]


def fetch_who(max_age_hours=24, force=False):
    """Download the CSV, reuse the cached copy if it is fresh, fall back to cache on network errors."""
    fresh = CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < max_age_hours * 3600
    if fresh and not force:
        return pd.read_csv(CACHE)
    try:
        r = requests.get(WHO_URL, timeout=60, headers={"User-Agent": "covid-forecast-app"})
        r.raise_for_status()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(r.content)
    except requests.RequestException as e:
        if not CACHE.exists():
            raise RuntimeError(f"Download failed and no cached copy exists: {e}")
        print(f"WARNING: download failed ({e}); using cached copy")
    df = pd.read_csv(CACHE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:                       # WHO changed the schema: fail loudly, not silently
        raise ValueError(f"WHO file is missing expected columns {missing}; found {list(df.columns)}")
    return df


def prepare(df):
    d = df[REQUIRED].copy()
    d["Date_reported"] = pd.to_datetime(d["Date_reported"])
    d[["New_cases", "New_deaths"]] = d[["New_cases", "New_deaths"]].fillna(0).clip(lower=0)  # WHO files contain negative corrections
    d = d.sort_values(["Country", "Date_reported"])
    g = d.groupby("Country")
    for L in range(4):
        d[f"lag_{L}"] = g["New_cases"].shift(L)
    d["death_lag_0"] = d["New_deaths"]
    d["growth"] = d["lag_0"] / (d["lag_1"] + 1)
    d["log_cum"] = np.log1p(d["Cumulative_cases"])
    d["week"] = d["Date_reported"].dt.isocalendar().week.astype(int)
    d["target"] = g["New_cases"].shift(-1)        # next week's cases
    return d


def X_of(d):
    X = d[FEATS].copy()
    X[["lag_0", "lag_1", "lag_2", "lag_3", "death_lag_0"]] = np.log1p(X[["lag_0", "lag_1", "lag_2", "lag_3", "death_lag_0"]])
    return X


def train(df, test_weeks=26):
    d = prepare(df).dropna(subset=FEATS + ["target"])
    d = d[d.Cumulative_cases >= 1000]
    cut = d.Date_reported.max() - pd.Timedelta(weeks=test_weeks)
    tr, te = d[d.Date_reported <= cut], d[d.Date_reported > cut]
    fit = lambda x: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42).fit(
        X_of(x), np.log1p(x.target) - np.log1p(x.lag_0))
    m = fit(tr)
    pred = np.expm1(m.predict(X_of(te)) + np.log1p(te.lag_0))
    metrics = {"MAE": float(mean_absolute_error(te.target, pred)),
               "baseline_MAE": float(mean_absolute_error(te.target, te.lag_0)),
               "test_rows": int(len(te)), "data_end": str(df.Date_reported.max())}
    joblib.dump({"model": fit(d), "metrics": metrics}, MODEL)
    return metrics


def predict_next_week(df, country):
    d = prepare(df)
    row = d[d.Country == country].dropna(subset=FEATS).tail(1)
    if row.empty:
        raise KeyError(country)
    art = joblib.load(MODEL)
    p = float(np.expm1(art["model"].predict(X_of(row))[0] + np.log1p(row.lag_0.iloc[0])))
    last = row.Date_reported.iloc[0]
    return {"country": country, "last_reported_week_ending": str(last.date()),
            "forecast_week_ending": str((last + pd.Timedelta(weeks=1)).date()),
            "last_week_cases": float(row.lag_0.iloc[0]), "predicted_next_week_cases": round(max(p, 0), 1)}


if __name__ == "__main__":
    data = fetch_who(force=True)
    print(train(data))
