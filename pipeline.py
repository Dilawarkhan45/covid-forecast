"""Data cleaning + feature engineering, shared by training and serving."""
import numpy as np
import pandas as pd

COUNTRY_MAP = {
    "Mainland China": "China", "('St. Martin',)": "St. Martin",
    "Bahamas, The": "Bahamas", "The Bahamas": "Bahamas",
    "Gambia, The": "Gambia", "The Gambia": "Gambia",
    "Republic of Ireland": "Ireland", "North Ireland": "UK",
    "Cape Verde": "Cabo Verde", "East Timor": "Timor-Leste",
    "occupied Palestinian territory": "Palestine", "West Bank and Gaza": "Palestine",
    "Holy See": "Vatican City", "Czech Republic": "Czechia",
    "Burma": "Myanmar", "Ivory Coast": "Cote d'Ivoire",
}
FEATURES = ["New_Confirmed_7d", "New_Deaths_7d", "lag_1", "lag_7", "lag_14", "lag_21",
            "growth_7", "death_lag_7", "log_confirmed", "log_active", "mortality", "dow"]
LOG_COLS = ["New_Confirmed_7d", "New_Deaths_7d", "lag_1", "lag_7", "lag_14", "lag_21", "death_lag_7"]
HORIZON = 7
MIN_CONFIRMED = 100


def clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ObservationDate"] = pd.to_datetime(d["ObservationDate"], format="%m/%d/%Y")
    d["Last Update"] = pd.to_datetime(d["Last Update"], format="mixed", errors="coerce")
    d["Country/Region"] = d["Country/Region"].str.strip().replace(COUNTRY_MAP)
    d["Province/State"] = d["Province/State"].fillna("Unknown")
    for c in ["Confirmed", "Deaths", "Recovered"]:
        d[c] = d[c].clip(lower=0).astype("int64")
    return (d.sort_values("Last Update")
             .drop_duplicates(["ObservationDate", "Province/State", "Country/Region"], keep="last")
             .drop(columns=["SNo"], errors="ignore").reset_index(drop=True))


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Active"] = (d["Confirmed"] - d["Deaths"] - d["Recovered"]).clip(lower=0)
    return d


def to_country_daily(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.groupby(["Country/Region", "ObservationDate"], as_index=False)
          [["Confirmed", "Deaths", "Recovered", "Active"]].sum()
          .sort_values(["Country/Region", "ObservationDate"]))
    for c in ["Confirmed", "Deaths", "Recovered"]:
        g["New_" + c] = g.groupby("Country/Region")[c].diff().fillna(g[c]).clip(lower=0)
    for c in ["New_Confirmed", "New_Deaths"]:
        g[c + "_7d"] = g.groupby("Country/Region")[c].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return g


def build_features(cd: pd.DataFrame, with_target: bool = False) -> pd.DataFrame:
    """Lag features per country. Rows with missing features are dropped."""
    d = cd.sort_values(["Country/Region", "ObservationDate"]).copy()
    g = d.groupby("Country/Region")
    for L in [1, 7, 14, 21]:
        d[f"lag_{L}"] = g["New_Confirmed_7d"].shift(L)
    d["growth_7"] = d["New_Confirmed_7d"] / (d["lag_7"] + 1)
    d["death_lag_7"] = g["New_Deaths_7d"].shift(7)
    d["log_confirmed"] = np.log1p(d["Confirmed"])
    d["log_active"] = np.log1p(d["Active"])
    d["mortality"] = np.where(d.Confirmed > 0, d.Deaths / d.Confirmed * 100, 0)
    d["dow"] = d.ObservationDate.dt.dayofweek
    need = FEATURES
    if with_target:
        d["target"] = g["New_Confirmed_7d"].shift(-HORIZON)
        need = FEATURES + ["target"]
    d = d.dropna(subset=need)
    return d[d.Confirmed >= MIN_CONFIRMED]


def model_matrix(d: pd.DataFrame) -> pd.DataFrame:
    X = d[FEATURES].copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    return X
