import pytest
import pandas as pd
import os
import sys
import pickle
from starter.ml.data import process_data
from starter.train_model import cat_features

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

@pytest.fixture
def clean_data():
    return pd.read_csv("./data/clean_data.csv")

def test_data_columns(clean_data):
    assert set(["age", "workclass", "fnlgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "salary"]).issubset(clean_data.columns)

def test_preprocess_data(clean_data):
    assert clean_data.duplicated().sum() == 0
    assert clean_data.isnull().sum().sum() == 0  # Ensure no missing values

def test_model_existence():
    assert os.path.isfile('./model/model.pkl')  # Ensure the filename matches

def test_model_inference():
    data = {"age": 48,
            "workclass": "Self-emp-inc",
            "fnlgt": 194924,
            "education": "HS-grad",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 39222,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States"
            }
    
    with open('./model/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('./model/lb.pkl', 'rb') as file:
        lb = pickle.load(file)
    with open('./model/model.pkl', 'rb') as file:
        model = pickle.load(file)

    data = pd.DataFrame(data, index=[0]) #`index=[0] allows to create a df from a scalar-valued dictionary
    processed_data, _, _, _ = process_data(
        data, categorical_features=cat_features, 
        label=None, training=False, encoder=encoder, lb=lb
    )

    predictions = model.predict(processed_data)
    assert all(pred in [0, 1] for pred in predictions)  # Check all predictions are either 0 or 1