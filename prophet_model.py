import pandas as pd
from prophet import Prophet

df = pd.read_csv("Forecasting Case- Study.xlsx - Sheet1.csv")

df["ds"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
df["y"] = pd.to_numeric(df["Total"].astype(str).str.replace(",", "", regex=False), errors="coerce")

df = df.dropna(subset=["ds", "y"]).groupby("ds", as_index=False)["y"].sum()

model = Prophet()

model.fit(df)

future = model.make_future_dataframe(periods=56)

forecast = model.predict(future)

print(forecast[['ds', 'yhat']].tail())
