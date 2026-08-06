import os
import pandas as pd
import joblib
import pytest

def test_dataset_generation():
    assert os.path.exists("datasets/train.csv")
    assert os.path.exists("datasets/test.csv")
    df = pd.read_csv("datasets/train.csv")
    assert len(df) > 0
    # check targets are not null
    assert df["Target"].isnull().sum() < len(df)

def test_model_loading():
    assert os.path.exists("saved_models/model.joblib")
    model = joblib.load("saved_models/model.joblib")
    assert model is not None
