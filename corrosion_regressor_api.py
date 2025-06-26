# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("corrosion_regressor_api")

# Create input/output pydantic models
input_model = create_model("corrosion_regressor_api_input", **{'NO2_holesovice': 0.30450183153152466, 'PM10_holesovice': 0.06942236423492432, 'O3_holesovice': 0.0, 'PM2_5_holesovice': 0.017499472945928574, 'temp_in': 0.2729138433933258, 'hum_in': 0.9808334708213806, 'dew_in': 0.5243192911148071, 'SO2': 0.07998329401016235, 'NO2': 0.13874365389347076, 'PM10': 0.04360372945666313, 'O3': 0.31511956453323364, 'PM2_5': 0.0067485361360013485, 'temp_out': 0.40951400995254517, 'dew_out': 0.5665801167488098, 'qnh': 0.37961655855178833, 'winddir': 0.6142338514328003, 'windspeed': 0.4801722764968872, 'hum_out': 0.8281739950180054, 'temp_in_2': 0.2688685655593872, 'hum_in_2': 0.9620082378387451, 'dew_in_2': 0.5105094313621521, 'temp_in_3': 0.2661465108394623, 'hum_in_3': 0.9918216466903687, 'dew_in_3': 0.5488617420196533})
output_model = create_model("corrosion_regressor_api_output", prediction=0.8)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
