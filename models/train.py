import os
import sys
import json
import pandas as pd
from lightgbm import LGBMClassifier
from shared.ml.pipeline import MLPipeline

def train_project_model():
    train_path = "datasets/train.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError("Missing datasets/train.csv. Run generate_data.py first.")
        
    df = pd.read_csv(train_path)
    
    numerical_features = ["Age", "MaxHeartRate", "RestBP", "Cholesterol", "ST_Depression"]
    categorical_features = ["Sex", "ChestPainType"]
    target_column = "Target"
    
    pipeline = MLPipeline(
        model_type="classification",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target_column
    )
    
    estimator = LGBMClassifier(random_seed=42)
    param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    
    metrics = pipeline.train_and_tune(df, estimator, param_grid)
    
    os.makedirs("saved_models", exist_ok=True)
    pipeline.save("saved_models/model.joblib")
    
    with open("saved_models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Model trained and serialized!")

if __name__ == "__main__":
    train_project_model()
