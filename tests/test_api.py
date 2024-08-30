from fastapi.testclient import TestClient
import os
import sys
import json

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from main import app

client = TestClient(app)

######## GET ###########

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome"


####### POST ###########

def test_predict_invalid():
    data = {}
    response = client.post("/predict/", json=json.dumps(data))
    assert response.status_code == 422

def test_predict_positive():
    data = {"age": 48,
            "workclass": "Self-emp-inc",
            "fnlgt": 194924,
            "education": "HS-grad",
            "education_num": 10,
            "marital_status": "Never-married",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 39222,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
            }
    response = client.post("/predict/", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {"pred": 1}


def test_predict_negative():
    data = {"age": 32,
            "workclass": "State-gov",
            "fnlgt": 67416,
            "education": "11th",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 1174,
            "capital_loss": 2310,
            "hours_per_week": 40,
            "native_country": "Cuba"
            }
    response = client.post("/predict/", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {"pred": 0}