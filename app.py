from fastapi import FastAPI
from pathlib import Path
import joblib
import numpy as np

app = FastAPI()

model_path = Path("best_model.pkl")
model = joblib.load(model_path) if model_path.exists() else None

@app.get("/")
def home():
    return {"message": "Sales Forecast API Running"}

@app.get("/predict")
def predict():
    sample_input = np.array([[100, 120, 140, 130, 10, 2, 5, 2026]])

    if model is None:
        return {"error": "Model file not found. Run main.py first to generate best_model.pkl."}

    prediction = model.predict(sample_input)

    return {"forecast": float(prediction[0])}
