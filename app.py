"""REST API.  Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import service
import live

app = FastAPI(title="COVID-19 Case Forecast API", version="1.0.0",
              description="Predicts a country's 7-day-average daily new cases, 7 days ahead.")


class Forecast(BaseModel):
    country: str
    as_of: str
    forecast_for: str
    current_daily_cases_7d_avg: float
    predicted_daily_cases_7d_avg: float
    total_confirmed: int


@app.get("/health")
def health():
    art, _, _ = service.load()
    return {"status": "ok", "data_end": art["data_end"], "metrics": art["metrics"]}


@app.get("/countries")
def list_countries():
    return {"countries": service.countries()}


@app.get("/predict", response_model=Forecast)
def predict(country: str = Query(..., examples=["India"]),
            as_of: str | None = Query(None, description="YYYY-MM-DD, defaults to latest date")):
    try:
        return service.predict(country, as_of)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/predict-weekly")
def predict_weekly(country: str = Query(..., examples=["India"])):
    """Next-week forecast from live WHO weekly data (run `python live.py` first)."""
    try:
        return live.predict_next_week(live.fetch_who(), country)
    except KeyError:
        raise HTTPException(404, f"Unknown country: {country}")
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(503, f"Live data/model not ready: {e}")
