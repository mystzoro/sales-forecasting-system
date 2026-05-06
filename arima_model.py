import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("Forecasting Case- Study.xlsx - Sheet1.csv")

df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
df["Sales"] = pd.to_numeric(df["Total"].astype(str).str.replace(",", "", regex=False), errors="coerce")

sales_data = df.dropna(subset=["Date", "Sales"]).groupby("Date")["Sales"].sum()

model = ARIMA(sales_data, order=(5,1,0))

model_fit = model.fit()

forecast = model_fit.forecast(steps=56)

print(forecast)
