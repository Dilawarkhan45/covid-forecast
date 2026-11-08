"""Model loading + prediction logic (used by both API and dashboard)."""
from functools import lru_cache
from pathlib import Path
import joblib, numpy as np, pandas as pd
from pipeline import build_features, model_matrix, HORIZON

BASE = Path(__file__).parent


@lru_cache(maxsize=1)
def load():
    art = joblib.load(BASE / "models/forecast_model.joblib")
    cd = pd.read_csv(BASE / "data/country_daily.csv", parse_dates=["ObservationDate"])
    feats = build_features(cd)  # no target needed: includes the latest dates
    return art, cd, feats


def countries():
    return sorted(load()[2]["Country/Region"].unique())


def predict(country: str, as_of: str | None = None) -> dict:
    art, cd, feats = load()
    rows = feats[feats["Country/Region"] == country]
    if rows.empty:
        raise KeyError(f"Unknown country or too few cases: {country}")
    if as_of:
        row = rows[rows.ObservationDate <= pd.Timestamp(as_of)].tail(1)
        if row.empty:
            raise ValueError(f"No data for {country} on or before {as_of}")
    else:
        row = rows.tail(1)
    x = model_matrix(row)
    log_pred = art["model"].predict(x)[0] + np.log1p(row["New_Confirmed_7d"].iloc[0])
    pred = max(float(np.expm1(log_pred)), 0.0)
    obs = row["ObservationDate"].iloc[0]
    return {"country": country,
            "as_of": str(obs.date()),
            "forecast_for": str((obs + pd.Timedelta(days=HORIZON)).date()),
            "current_daily_cases_7d_avg": round(float(row["New_Confirmed_7d"].iloc[0]), 1),
            "predicted_daily_cases_7d_avg": round(pred, 1),
            "total_confirmed": int(row["Confirmed"].iloc[0])}


def history(country: str, days: int = 120) -> pd.DataFrame:
    cd = load()[1]
    return cd[cd["Country/Region"] == country].tail(days)[["ObservationDate", "New_Confirmed", "New_Confirmed_7d"]]
