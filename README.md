# COVID-19 Forecast: deployment package

Two models:
1. **Daily model (historical demo)**: predicts a country's 7-day-average daily new cases, 7 days ahead. Trained on `covid_19_data.csv` (JHU-style, ends 2021-05-29).
2. **Weekly model (live data)**: predicts next week's new cases from WHO weekly data via `live.py`.

## Quick start
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py            # daily model -> models/forecast_model.joblib (needs data/covid_19_data.csv)
python live.py            # weekly model -> downloads WHO CSV, trains, prints MAE vs baseline
pytest -q
uvicorn app:app --port 8000          # API docs: http://localhost:8000/docs
streamlit run streamlit_app.py       # dashboard: http://localhost:8501
```

## API
- `GET /health`
- `GET /countries`
- `GET /predict?country=India[&as_of=2021-04-20]`   (daily model, historical)
- `GET /predict-weekly?country=India`               (weekly model, live WHO data; 503 until `python live.py` has run)

## Live data notes
- JHU stopped updating on 2023-03-10. Current data = WHO weekly file:
  https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv
- WHO country names differ from the JHU ones (use `/predict-weekly` with WHO spellings).
- After `python live.py`, compare `MAE` with `baseline_MAE`. Deploy the weekly model only if it beats the baseline.
- `.github/workflows/refresh.yml` retrains weekly on GitHub Actions (untested).
- `live.py` expects the columns Date_reported, Country, New_cases, New_deaths, Cumulative_cases and raises a clear error if WHO changes them.

## Docker
```bash
docker build -t covid-forecast .
docker run -p 8000:8000 covid-forecast
docker run -p 8501:8501 covid-forecast streamlit run streamlit_app.py --server.address 0.0.0.0
```

## Deploy
- **Dashboard:** Streamlit Community Cloud, main file `streamlit_app.py` (commit `models/` and `data/country_daily.csv`).
- **API:** Render / Railway / Fly.io / Cloud Run, using the Dockerfile.

## Caveats
Daily-model data stops in May 2021. Daily-model held-out MAE ~532 cases/day vs 637 naive baseline. Current WHO reporting is patchy and undercounts infections; not for medical or policy decisions.
