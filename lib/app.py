# Save this as app.py and run via `streamlit run app.py`
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Bike Sharing Seasonality & Demand Analysis")

df = pd.read_csv("./data/cleaned.csv")

# Interactive slider for the Moving Average window
window = st.slider("Select Moving Average Window", min_value=3, max_value=30, value=7)
df['Custom_MA'] = df['cnt'].shift(1).rolling(window=window).mean()

# Plotly interactive chart
fig = px.line(df, y=['cnt', 'Custom_MA'], title="Actual Demand vs Custom Moving Average")
st.plotly_chart(fig)