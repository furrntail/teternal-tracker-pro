import streamlit as st
import pandas as pd
from predictor import make_prediction
from backtest import run_backtest

st.title("📈 Eternal Ltd. Stock Predictor")

uploaded_file = st.file_uploader("Upload 1-Hour Interval CSV (timestamp, open, high, low, close, volume)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    st.write("### Uploaded Data", df.tail())

    if st.button("Predict Next Hour Trend"):
        prediction, confidence = make_prediction(df)
        st.success(f"Prediction: {prediction} (Confidence: {confidence}%)")

    if st.button("Run Backtest"):
        results = run_backtest(df)
        st.write("### Backtest Results", results)