import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

df = pd.read_csv("Forecasting Case- Study.xlsx - Sheet1.csv")

# Normalize the provided dataset into a single daily sales series.
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
df["Sales"] = pd.to_numeric(df["Total"].astype(str).str.replace(",", "", regex=False), errors="coerce")

df = df.dropna(subset=["Date", "Sales"]).sort_values("Date")
df = df.groupby("Date", as_index=False)["Sales"].sum()

df.ffill(inplace=True)

df["lag_1"] = df["Sales"].shift(1)
df["lag_7"] = df["Sales"].shift(7)
df["lag_30"] = df["Sales"].shift(30)

df["rolling_mean_7"] = df["Sales"].rolling(7).mean()
df["rolling_std_7"] = df["Sales"].rolling(7).std()

df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year

df.dropna(inplace=True)

train_df = df[:-56]
test_df = df[-56:]

feature_cols = [
	"lag_1",
	"lag_7",
	"lag_30",
	"rolling_mean_7",
	"rolling_std_7",
	"day_of_week",
	"month",
	"year",
]

X_train = train_df[feature_cols]
y_train = train_df["Sales"]

X_test = test_df[feature_cols]
y_test = test_df["Sales"]

model = XGBRegressor(
	n_estimators=100,
	learning_rate=0.1,
	max_depth=5,
	random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Sales Forecasting")
plt.show()

joblib.dump(model, "best_model.pkl")

print("Model Saved")
