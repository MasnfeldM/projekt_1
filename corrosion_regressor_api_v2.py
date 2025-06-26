# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import io
import requests
import os
import plotly.graph_objs as go
import plotly.io as pio

# Vytvoření FastAPI aplikace
app = FastAPI()

# Načtení trénovaného modelu
model = load_model("corrosion_regressor_model")

# Cesta k uloženým CSV souborům
UPLOAD_PATH = "uploaded_data.csv"
PREDICTION_PATH = "predictions.csv"

# Upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(None), url: str = Form(None)):
    try:
        if file:
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif url:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        else:
            return JSONResponse(content={"error": "Please provide a file or URL."}, status_code=400)
        
        df = df.drop(columns=["corrosion", "corrosion_diff"], errors='ignore') 
        df.to_csv(UPLOAD_PATH, index=False)
        return {"message": "Data uploaded successfully", "rows": len(df)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Predict endpoint
@app.get("/predict")
def predict():
    if not os.path.exists(UPLOAD_PATH):
        return JSONResponse(content={"error": "No uploaded data found."}, status_code=404)

    try:
        df = pd.read_csv(UPLOAD_PATH)
        df = df.drop(columns=["Datetime"])

        predictions = predict_model(model, data=df)
        predictions = predictions["prediction_label"]
        predictions.to_csv(PREDICTION_PATH, index=False)

        return {"message": "Predictions generated successfully", "rows": len(predictions), "file": PREDICTION_PATH}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Endpoint pro vytvoření grafu
@app.get("/make_graph", response_class=HTMLResponse)
def make_graph():
    if not os.path.exists(UPLOAD_PATH):
        return JSONResponse(content={"error": "No uploaded data found."}, status_code=404)

    try:
        df = pd.read_csv(UPLOAD_PATH)
        predictions = pd.read_csv(PREDICTION_PATH)
        datetime = df["Datetime"] if "Datetime" in df.columns else df.index
        datetime = pd.to_datetime(datetime)
        y_points = predictions["prediction_label"]
        # Vytvoření grafu (např. index vs predikce)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=y_points,
                mode='lines',
                name='Odhad koroze'
            )
)
        # Vygenerování HTML kódu grafu
        html = pio.to_html(fig, full_html=True)
        return HTMLResponse(content=html)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Spuštění aplikace
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
