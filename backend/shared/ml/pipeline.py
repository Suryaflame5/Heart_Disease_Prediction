import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from shared.ml.metrics import calculate_regression_metrics, calculate_classification_metrics

class MLPipeline:
    def __init__(
        self,
        model_type: str,  # 'regression' or 'classification' or 'clustering'
        numerical_features: List[str],
        categorical_features: List[str],
        target_column: str = None
    ):
        self.model_type = model_type
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.pipeline = None

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        return preprocessor

    def train_and_tune(
        self,
        df: pd.DataFrame,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        # Handle missing target
        if self.target_column:
            # Drop rows where target is missing
            df = df.dropna(subset=[self.target_column])
            X = df[self.numerical_features + self.categorical_features]
            y = df[self.target_column]
        else:
            X = df[self.numerical_features + self.categorical_features]
            y = None

        preprocessor = self.build_preprocessor()

        if self.model_type == 'clustering':
            # Unsupervised: no train/test split needed for clustering
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('estimator', estimator)
            ])
            pipeline.fit(X)
            self.pipeline = pipeline
            
            # Evaluate using Silhouette Score if clustering
            from sklearn.metrics import silhouette_score
            X_trans = preprocessor.transform(X)
            labels = pipeline.named_steps['estimator'].labels_
            
            # Handle edge case where all labels are same
            if len(np.unique(labels)) > 1:
                sil = float(silhouette_score(X_trans, labels))
            else:
                sil = 0.0
                
            return {
                "silhouette_score": sil,
                "labels": labels.tolist()
            }

        # For supervised learning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Build full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])

        # Hyperparameter tuning using GridSearchCV
        # Prefix the parameter grid keys with 'estimator__'
        prefixed_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
        
        # Determine scoring
        scoring = 'r2' if self.model_type == 'regression' else 'f1_macro'

        grid_search = GridSearchCV(
            full_pipeline,
            prefixed_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            error_score='raise'
        )

        grid_search.fit(X_train, y_train)
        
        self.pipeline = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        
        if self.model_type == 'regression':
            metrics = calculate_regression_metrics(y_test.values, y_pred)
        else:
            # Classification
            try:
                y_prob = self.pipeline.predict_proba(X_test)
            except AttributeError:
                y_prob = None
            metrics = calculate_classification_metrics(y_test.values, y_pred, y_prob)

        metrics["best_params"] = {k.replace("estimator__", ""): v for k, v in grid_search.best_params_.items()}
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise ValueError("Pipeline has not been trained yet.")
        return self.pipeline.predict(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise ValueError("Pipeline has not been trained yet.")
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(df)
        raise AttributeError("This model does not support class probability estimation.")

    def save(self, file_path: str):
        if not self.pipeline:
            raise ValueError("No trained pipeline to save.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'MLPipeline':
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        return joblib.load(file_path)
