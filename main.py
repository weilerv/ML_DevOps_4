# Put the code for your API here.
'''The API must implement GET and POST. GET must be on the root domain and 
give a greeting and POST on a different path that does model inference.

Use Python type hints such that FastAPI creates the automatic documentation.

Use a Pydantic model to ingest the body of the POST. This should implement 
an example (hint: Pydantic/FastAPI provides multiple ways to do this, see the docs for more information: https://fastapi.tiangolo.com/tutorial/schema-extra-example/(opens in a new tab)).

Include a screenshot of the docs that shows the example and name it example.png.'''

import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import os 
import pickle
import numpy as np

from ml.data import process_data
from ml.model import inference

#load model
model_path = 'model/trained_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
encoder_path = 'model/encoder.pkl'
with open(encoder_path, 'rb') as file2:
    encoder = pickle.load(file2)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

app = FastAPI()

#Define get
@app.get('/')
async def greeting():
    return {"greeting": "Welcome to my salary prediction api"}

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str	 
    race: str 
    sex: str
    capital_gain: int = Field(alias='capital-gain') 
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }
@app.post('/prediction')
async def predict(data: InputData):
    data = dict(data)
    vals = np.array(list(data.values()))
    vals = vals.reshape((1, 14))
    df = pd.DataFrame(columns=data.keys(), data=vals)
    #df = pd.DataFrame.from_dict(data)
    pred_data, _, _, _ = process_data(df,
                                      categorical_features=cat_features,
                                      training=False,
                                      encoder=encoder)
    prediction = inference(model, pred_data)
    prediction = int(prediction)
    pred = {'<=50K' if prediction == 1 else '>50K'}
    return {'prediction:' : pred}