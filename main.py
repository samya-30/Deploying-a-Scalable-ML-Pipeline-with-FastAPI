import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Pydantic schema
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Load model and encoder
encoder = load_model("model/encoder.pkl")
model = load_model("model/model.pkl")

# Create FastAPI app
app = FastAPI()

@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Welcome to the Income Prediction API!"}

@app.post("/data/")
async def post_inference(data: Data):
    # Convert Pydantic model to dict
    data_dict = data.dict()
    # Rename fields to match original dataset format
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder)
    pred = inference(model, X)
    return {"result": apply_label(pred)}
