from fastapi.testclient import TestClient
import os
try:
    from main import app
except:
    from .main import app

client = TestClient(app)

def test_get():
    request = client.get("/")
    assert request.status_code == 200, 'response failed'
    assert request.json() == {"greeting": "Welcome to my salary prediction api"}

def test_post_one():
    input_data = {'age': 50,
                  'workclass': 'Self-emp-inc',
                  'fnlgt': 83311,
                  'education': 'Bachelors',
                  'education-num': 13,
                  'marital-status': 'Married-civ-spouse',
                  'occupation': 'Exec-managerial',
                  'relationship': 'Husband',
                  'race': 'White',
                  'sex': 'Male',
                  'capital-gain': 0,
                  'capital-loss': 0,
                  'hours-per-week': 13,
                  'native-country': 'United-States'
                  }
    response = client.post('/prediction/', json=input_data)

    assert response.status_code == 200, f'Status code {response.json()} returned instead of 200'
    assert response.json() == {"prediction:": ["<=50K"]}, "wrong prediction: expected <=50K, but the result was >50K"

def test_post_two():
    input_data = {'age': 52,
                  'workclass': 'Self-emp-not-inc',
                  'fnlgt': 209642,
                  'education': 'HS-grad',
                  'education-num': 9,
                  'marital-status': 'Married-civ-spouse',
                  'occupation': 'Exec-managerial',
                  'relationship': 'Husband',
                  'race': ' White',
                  'sex': ' Male',
                  'capital-gain': 0,
                  'capital-loss': 0,
                  'hours-per-week': 45,
                  'native-country': 'United-States'
                  }
    response = client.post('/prediction/', json=input_data)

    assert response.status_code == 200, f'Status code {response.json()} returned instead of 200'
    assert response.json() == {"prediction:": [">50K"]}, "wrong prediction: expected >50K, but the result is <=50K"
