from fastapi.testclient import TestClient
from app import app

c = TestClient(app)


def test_health():
    r = c.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_latest():
    r = c.get("/predict", params={"country": "India"}); assert r.status_code == 200
    j = r.json(); assert j["predicted_daily_cases_7d_avg"] >= 0 and j["forecast_for"] > j["as_of"]


def test_predict_as_of():
    r = c.get("/predict", params={"country": "US", "as_of": "2021-01-10"})
    assert r.status_code == 200 and r.json()["as_of"] == "2021-01-10"


def test_unknown_country():
    assert c.get("/predict", params={"country": "Atlantis"}).status_code == 404


def test_before_data():
    assert c.get("/predict", params={"country": "US", "as_of": "2019-01-01"}).status_code == 422


def test_weekly_not_ready_or_ok():
    # 503 if live.py has not been run yet, 200 after it has; never a crash
    assert c.get("/predict-weekly", params={"country": "India"}).status_code in (200, 404, 503)
