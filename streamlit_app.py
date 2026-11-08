"""Dashboard.  Run: streamlit run streamlit_app.py"""
import pandas as pd
import streamlit as st
import service

st.set_page_config(page_title="COVID-19 Forecast", layout="wide")
st.title("COVID-19 7-day case forecast")
art, cd, _ = service.load()

country = st.sidebar.selectbox("Country", service.countries(),
                               index=service.countries().index("India") if "India" in service.countries() else 0)
days = st.sidebar.slider("History shown (days)", 30, 300, 120)
res = service.predict(country)

c1, c2, c3 = st.columns(3)
c1.metric("Current daily cases (7-day avg)", f"{res['current_daily_cases_7d_avg']:,.0f}")
c2.metric(f"Predicted for {res['forecast_for']}", f"{res['predicted_daily_cases_7d_avg']:,.0f}",
          f"{res['predicted_daily_cases_7d_avg'] - res['current_daily_cases_7d_avg']:+,.0f}")
c3.metric("Total confirmed", f"{res['total_confirmed']:,}")

h = service.history(country, days).set_index("ObservationDate")
st.line_chart(h.rename(columns={"New_Confirmed": "Daily new", "New_Confirmed_7d": "7-day avg"}))
st.caption(f"Data ends {art['data_end']}. Held-out MAE: {art['metrics']['MAE']:,.0f} cases/day "
           f"(naive baseline {art['metrics']['baseline_MAE']:,.0f}). Not for medical or policy decisions.")
