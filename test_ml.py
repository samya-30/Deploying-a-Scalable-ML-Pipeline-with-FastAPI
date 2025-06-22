import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Sample data for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame([{
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K"
    }])

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

def test_process_data(sample_data):
    """Test that process_data returns the correct shapes and types."""
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    assert X.shape[0] == 1
    assert isinstance(y[0], (int, float, np.integer, np.floating))

def test_train_model(sample_data):
    """Test that a model can be trained on the sample data."""
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == 1

def test_model_metrics(sample_data):
    """Test that model metrics return valid float scores."""
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f1 = compute_model_metrics(y, preds)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f1, float)
