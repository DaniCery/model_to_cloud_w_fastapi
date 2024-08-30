# Put the code for your API here.
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field, field_validator
from enum import Enum
import pickle
import os
import uvicorn
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import cat_features

# Istantiate the FastAPI app
app = FastAPI(
    title="Model to Cloud with FastAPI",
    description="An API that demonstrates classificaiton of census bureau data in assessing salary above or under 50k$",
    version="1.0.0",
)

# Load model
file_dir = os.path.dirname(__file__)
model_path = os.path.join(file_dir, './model/model.pkl')
encoder_path = os.path.join(file_dir, './model/encoder.pkl')
lb_path = os.path.join(file_dir, './model/lb.pkl')

model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))


class Workclass(str, Enum):
    P = "Private"
    SEN = "Self-emp-not-inc"
    LG = "Local-gov"
    SG = "State-gov"
    SEI = "Self-emp-inc"
    FG = "Federal-gov"
    WP= "Without-pay"
    NW = "Never-worked"

class Data(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(gt=0, lt=122, example=39)
    workclass: Workclass = Field(example='State-gov')
    fnlgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(alias='education-num', example=13)
    marital_status: str = Field(alias='marital-status', example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Female')
    capital_gain: int = Field(ge=0, alias='capital-gain', example=2174)
    capital_loss: int = Field(ge=0, alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(alias='native-country',example='United-States')

    @field_validator('sex')
    def sex_must_be_male_or_female(cls, v):
        if v not in ("Male", "Female"):
            raise ValueError('Sex must be Male or Female.')
        return v
    
    # This allows the API to accept the original field names (with hyphens) in the JSON body of the POST request.
    # FastAPI will automatically map these to the corresponding Python field names.
    class Config:
        allow_population_by_field_name = True

@app.get("/")
async def greetings():
    return f'Welcome'

@app.post("/predict/")
async def predict(data: Data):
    if data.hours_per_week > 168:
        raise HTTPException(status_code=400, detail="hours_per_week needs to be lower than 168.")
    
    # Convert data to a DataFrame
    data = pd.DataFrame.from_records([data.model_dump()])
    
    # Manage hypens
    data.columns = [col.replace('_', '-') for col in data.columns]

    X, _, _, _ = process_data(data, categorical_features=cat_features, 
                    label=None, training=False, 
                    encoder=encoder, lb=lb)
    pred = inference(model, X)

    out = ['<=50K' if p == 0 else '>50K' for p in pred]

    return {"message": "Predictions", "pred": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)