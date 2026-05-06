import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("Forecasting Case- Study.xlsx - Sheet1.csv")

df['Sales'] = pd.to_numeric(df['Total'].astype(str).str.replace(',', '', regex=False), errors='coerce')
sales = df.dropna(subset=['Sales'])['Sales'].values.reshape(-1,1)

scaler = MinMaxScaler()

sales_scaled = scaler.fit_transform(sales)

X = []
y = []

for i in range(30, len(sales_scaled)):
    X.append(sales_scaled[i-30:i])
    y.append(sales_scaled[i])

X = np.array(X)
y = np.array(y)

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(30,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=5, batch_size=32)

print("LSTM Training Completed")
