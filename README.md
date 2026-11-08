# COVID-19 Case Forecasting: API and Dashboard

A machine-learning project that cleans COVID-19 case data, forecasts new cases per country, and serves the forecasts through a REST API and an interactive Streamlit dashboard.

It contains two models:

| Model            | Data                                                                                      | Predicts                                                | Status                                                          |
| ---------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| **Daily model**  | Kaggle "Novel Corona Virus 2019 Dataset", `covid_19_data.csv` (2020-01-22 to 2021-05-29)) | A country's 7-day-average daily new cases, 7 days ahead | Trained and tested                                              |
| **Weekly model** | WHO weekly global data, downloaded live by `live.py`                                      | A country's new cases for the next week                 | Code included; check its metrics after you run `python live.py` |

> **Data note:** The Johns Hopkins feed stopped updating on 10 March 2023, so the daily model is a historical demonstration. The live path uses the WHO weekly file instead.

---

## Features

- **Data cleaning:** parses dates, standardizes inconsistent country names (for example "Mainland China" to "China"), fixes negative counts, removes duplicate records, and fills missing provinces.
- **Feature engineering:** active cases, mortality and recovery rates, daily new cases from cumulative totals, 7-day rolling averages, and lag features (1, 7, 14, 21 days).
- **Modelling:** Ridge, Random Forest and Histogram Gradient Boosting compared against a naive baseline, using a **time-based split** so the model never sees the future.
- **REST API:** FastAPI with automatic interactive docs.
- **Dashboard:** Streamlit app with country selector, current versus predicted cases, and history chart.
- **Live data:** `live.py` downloads and caches the WHO weekly CSV, validates its columns, and retrains.
- **Tests and packaging:** pytest suite, Dockerfile, and a scheduled GitHub Actions refresh workflow.

---

## Dataset

The daily model is trained on `covid_19_data.csv` from the Kaggle **Novel Corona Virus 2019 Dataset**, which compiles case counts from the Johns Hopkins CSSE feed.

- **Source:** [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
- **File used:** `covid_19_data.csv` (306,429 rows, 22 January 2020 to 29 May 2021)

## Results (daily model)

Evaluated on data from 2021-03-01 onward (15,246 rows). The model was trained only on earlier data (57,994 rows).

| Model                                      | MAE       | RMSE    | R²    |
| ------------------------------------------ | --------- | ------- | ----- |
| Baseline (assume today's level continues)  | 636.7     | 4,275.9 | 0.949 |
| Ridge                                      | 600.3     | 3,870.5 | 0.958 |
| Random Forest                              | 648.6     | 6,712.9 | 0.874 |
| **Histogram Gradient Boosting (deployed)** | **532.3** | 3,931.8 | 0.957 |

MAE is in cases per day. The deployed model's error is about **16% lower** than the baseline. This is a modest improvement, because next week's cases mostly continue this week's trend. Large surges in big countries are still predicted poorly.

**Why the target is a growth ratio:** Predicting raw counts made the tree models fail on India's spring 2021 wave, which was larger than anything in the training period. Predicting the log growth ratio (next week relative to now) removes that limit.

---

## Project structure

```
covid-forecast/
├── app.py                  # FastAPI service
├── streamlit_app.py        # Streamlit dashboard
├── service.py              # Model loading and prediction logic
├── pipeline.py             # Cleaning + feature engineering (shared by train and serve)
├── train.py                # Trains the daily model
├── live.py                 # WHO weekly data: download, train, predict
├── test_api.py             # API tests
├── requirements.txt
├── Dockerfile
├── data/
│   ├── covid_19_data.csv   # Raw data (Kaggle Novel Corona Virus 2019 Dataset)
│   └── country_daily.csv   # Cleaned country-level daily data
├── models/
│   ├── forecast_model.joblib
│   └── metrics.json
└── .github/workflows/
    └── refresh.yml         # Weekly data refresh
```

---

## Quick start

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/Dilawarkhan45/covid-forecast.git
cd covid-forecast

python -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1

pip install -r requirements.txt
# Download covid_19_data.csv from the Kaggle dataset and place it in data/
python train.py                   # builds data/country_daily.csv and the daily model
pytest -q                         # expect all tests to pass
```

Run the API:

```bash
uvicorn app:app --port 8000
```

Interactive docs: http://localhost:8000/docs

Run the dashboard (in a second terminal, with the venv activated):

```bash
streamlit run streamlit_app.py
```

Dashboard: http://localhost:8501

---

## API reference

| Endpoint                                   | Description                                     |
| ------------------------------------------ | ----------------------------------------------- |
| `GET /health`                              | Service status, data end date and model metrics |
| `GET /countries`                           | List of supported countries                     |
| `GET /predict?country=India`               | Daily-model forecast from the latest date       |
| `GET /predict?country=US&as_of=2021-01-10` | Forecast as of a past date (`YYYY-MM-DD`)       |
| `GET /predict-weekly?country=India`        | Weekly-model forecast from live WHO data        |

Example response from `/predict?country=India`:

```json
{
  "country": "India",
  "as_of": "2021-05-29",
  "forecast_for": "2021-06-05",
  "current_daily_cases_7d_avg": 194952.6,
  "predicted_daily_cases_7d_avg": 160277.4,
  "total_confirmed": 27894800
}
```

Errors: `404` for an unknown country, `422` for a date before the country's data begins, and `503` from `/predict-weekly` if `python live.py` has not been run yet.

---

## Using live data (WHO)

```bash
python live.py
```

This downloads the WHO weekly file, caches it in `data/who_weekly.csv`, trains the weekly model, and prints the error alongside a "same as last week" baseline:

```
{'MAE': ..., 'baseline_MAE': ..., 'test_rows': ..., 'data_end': ...}
```

Only rely on the weekly model if `MAE` is lower than `baseline_MAE`.

- **Source:** `https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv`
- **Expected columns:** `Date_reported`, `Country`, `New_cases`, `New_deaths`, `Cumulative_cases`. If WHO changes these, `live.py` stops with a clear error instead of producing wrong results.
- **Country names** follow WHO's spelling and differ from the older Johns Hopkins names.
- The weekly refresh workflow in `.github/workflows/refresh.yml` retrains every Tuesday. It needs "Read and write" workflow permission in the repository settings.

---

## Docker

```bash
docker build -t covid-forecast .
docker run -p 8000:8000 covid-forecast
```

Dashboard instead of the API:

```bash
docker run -p 8501:8501 covid-forecast streamlit run streamlit_app.py --server.address 0.0.0.0
```

The image runs `train.py` during the build, so `data/covid_19_data.csv` must be present.

---

## Deployment

- **Dashboard:** [Streamlit Community Cloud](https://share.streamlit.io). Choose this repository and set the main file to `streamlit_app.py`.
- **API:** Any Docker host (Render, Railway, Fly.io, Google Cloud Run), pointing at the `Dockerfile`.
- Pin your `scikit-learn` version in `requirements.txt` to the version that trained the model, because saved model files can fail to load on a different version.

Live demo: _add your Streamlit URL here after deploying_

---

## Limitations

- The daily model's data ends on 2021-05-29, so "latest" forecasts are for early June 2021.
- Case counts reflect testing and reporting, not true infections. Reporting is patchy and revised after the fact, especially in recent WHO data.
- The `Recovered` column in the historical data is unreliable for some countries (the US reports 0), so Active and Recovery figures there are unreliable.
- The gain over the naive baseline is modest, and errors are larger for countries with sudden surges.
- Not for medical, public-health or policy decisions.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, FastAPI, Uvicorn, Streamlit, joblib, pytest, Docker, GitHub Actions.

---

## Author

**Dilawar** ([@Dilawarkhan45](https://github.com/Dilawarkhan45))
