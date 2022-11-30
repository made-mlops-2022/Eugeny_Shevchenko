import pytest
from fastapi.testclient import TestClient

from main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == 'Model ready'


def test_predict_endpoint():
    data = {
        'age': 59,
        'sex': 1,
        'cp': 0,
        'trestbps': 134,
        'chol': 204,
        'fbs': 0,
        'restecg': 0,
        'thalach': 162,
        'exang': 0,
        'oldpeak': 0.8,
        'slope': 0,
        'ca': 2,
        'thal': 0
    }
    response = client.post(
        '/predict',
        json=data
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'healthy'}


def test_missing_fields():
    data = {
        'age': 59,
        'sex': 1,
        'cp': 0,
        'fbs': 0,
        'thalach': 162,
        'exang': 0,
        'thal': 0
    }
    response = client.post(
        '/predict',
        json=data
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'


def test_wrong_categorical_fields():
    data = {
        'age': 59,
        'sex': 3,
        'cp': 0,
        'trestbps': 134,
        'chol': 204,
        'fbs': 0,
        'restecg': 0,
        'thalach': 162,
        'exang': 0,
        'oldpeak': 0.8,
        'slope': 0,
        'ca': 2,
        'thal': 0
    }
    response = client.post(
        '/predict',
        json=data
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'


def test_wrong_age_numerical_fields():
    data = {
        'age': 2022,
        'sex': 1,
        'cp': 0,
        'trestbps': 134,
        'chol': 204,
        'fbs': 0,
        'restecg': 0,
        'thalach': 162,
        'exang': 0,
        'oldpeak': 0.8,
        'slope': 0,
        'ca': 2,
        'thal': 0
    }
    response = client.post(
        '/predict',
        json=data
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'wrong age value'
