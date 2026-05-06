from fastapi import FastAPI
from pathlib import Path
import joblib

app = FastAPI()

model_path = Path("best_model.pkl")
model = joblib.load(model_path) if model_path.exists() else None

@app.get("/")
def home():
    return {
        "message": "Sales Forecast API Running"
    }

@app.get("/predict")
def predict():
    prediction = 5000

    return {
        "forecast": prediction
    }
