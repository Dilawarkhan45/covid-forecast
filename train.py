"""Train the forecasting model and save artifacts.  Usage: python train.py [csv_path]"""
import sys, json
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline import clean, add_features, to_country_daily, build_features, model_matrix, FEATURES

path = sys.argv[1] if len(sys.argv) > 1 else "data/covid_19_data.csv"
cd = pd.read_csv(path).pipe(clean).pipe(add_features).pipe(to_country_daily)
cd.to_csv("data/country_daily.csv", index=False)

d = build_features(cd, with_target=True)
cut = pd.Timestamp("2021-03-01")
tr, te = d[d.ObservationDate < cut], d[d.ObservationDate >= cut]
base = lambda x: np.log1p(x.New_Confirmed_7d.values)

def fit(df):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42)
    m.fit(model_matrix(df), np.log1p(df.target.values) - base(df))
    return m

# 1) honest evaluation on held-out period
m = fit(tr)
pred = np.expm1(m.predict(model_matrix(te)) + base(te))
metrics = {"MAE": float(mean_absolute_error(te.target, pred)),
           "RMSE": float(np.sqrt(mean_squared_error(te.target, pred))),
           "R2": float(r2_score(te.target, pred)),
           "baseline_MAE": float(mean_absolute_error(te.target, te.New_Confirmed_7d)),
           "train_rows": int(len(tr)), "test_rows": int(len(te))}
print(metrics)

# 2) final model fitted on all labelled data
final = fit(d)
joblib.dump({"model": final, "features": FEATURES, "metrics": metrics,
             "data_end": str(cd.ObservationDate.max().date())}, "models/forecast_model.joblib")
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print("Saved models/forecast_model.joblib")
