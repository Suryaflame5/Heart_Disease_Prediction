import os
import sys
import time
import io
import json
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List

# Setup path for shared library imports
sys.path.append(os.path.dirname(__file__))
from shared.backend.logger import setup_logger
from shared.backend.database import PredictionHistoryDB
from shared.backend.exceptions import (
    APIError, api_error_handler, validation_exception_handler, generic_exception_handler
)
from backend.schemas import PredictionInput, PredictionResponse, HistoryLog

# Structured logging
logger = setup_logger("06_Heart_Disease_Prediction")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Microservice for ECG Monitor prediction models.",
    version="1.0.0"
)

# Exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Database connection
db = PredictionHistoryDB(db_path=os.path.join(os.path.dirname(__file__), "history.db"))

# Lazy load model
model_pkg = None
try:
    model_pkg = joblib.load(os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.joblib"))
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")

# Recommendation cosine imports
from sklearn.metrics.pairwise import linear_kernel

@app.on_event("startup")
def startup_event():
    global model_pkg
    if model_pkg is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.joblib")
            if os.path.exists(model_path):
                model_pkg = joblib.load(model_path)
                logger.info("Model loaded at startup.")
        except Exception as e:
            logger.error(f"Error loading model at startup: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    global model_pkg
    if model_pkg is None:
        raise APIError("Model is not loaded or trained yet.", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        
        start_time = time.time()
        df_in = pd.DataFrame([data.dict()])
        pred = model_pkg.predict(df_in)
        pred_val = float(pred[0]) if isinstance(pred[0], (float, np.float32, np.float64)) else int(pred[0])
        prob_val = None
        if hasattr(model_pkg, "predict_proba"):
            try:
                probs = model_pkg.predict_proba(df_in)[0]
                prob_val = [float(p) for p in probs]
            except Exception:
                pass
        latency = (time.time() - start_time) * 1000
        result_payload = {"val": pred_val, "probabilities": prob_val}
        db.log_prediction("__PROJECT_ID__", data.dict(), result_payload, latency)
        return {"success": True, "prediction": result_payload, "latency_ms": latency}
        
    except Exception as e:
        logger.error(f"Prediction failure: {str(e)}")
        raise APIError(f"Failed to make prediction: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    global model_pkg
    if model_pkg is None:
        raise APIError("Model is not loaded or trained yet.", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    try:
        contents = await file.read()
        contents = contents.decode("utf-8")
        
        df = pd.read_csv(io.StringIO(contents))
        preds = model_pkg.predict(df)
        df["Prediction"] = [float(p) if isinstance(p, (float, np.float32, np.float64)) else int(p) for p in preds]
        results = df["Prediction"].tolist()
        
        
        # Write to memory buffer and stream back
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response_bytes = stream.getvalue().encode("utf-8")
        
        return StreamingResponse(
            io.BytesIO(response_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        logger.error(f"Batch prediction failure: {str(e)}")
        raise APIError(f"Failed to run batch predictions: {str(e)}")

@app.get("/history", response_model=List[HistoryLog])
def get_prediction_history(limit: int = 100):
    try:
        return db.get_history("06_Heart_Disease_Prediction", limit=limit)
    except Exception as e:
        logger.error(f"History retrieval failure: {str(e)}")
        raise APIError("Failed to fetch logs")

@app.post("/clear-history")
def clear_prediction_history():
    try:
        db.clear_history("06_Heart_Disease_Prediction")
        return {"success": True, "message": "History cleared"}
    except Exception as e:
        logger.error(f"History clear failure: {str(e)}")
        raise APIError("Failed to clear logs")

@app.get("/metrics")
def get_model_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found. Train the model first.")
    with open(metrics_path, "r") as f:
        return json.load(f)

@app.post("/train")
def train_model():
    try:
        import subprocess
        train_script = os.path.join(os.path.dirname(__file__), "..", "models", "train.py")
        result = subprocess.run([sys.executable, train_script], capture_output=True, text=True, check=True)
        
        # Reload model
        global model_pkg
        model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.joblib")
        model_pkg = joblib.load(model_path)
        
        logger.info("Model retrained successfully.")
        
        with open(os.path.join(os.path.dirname(__file__), "..", "saved_models", "metrics.json"), "r") as f:
            new_metrics = json.load(f)
            
        return {"success": True, "message": "Model retrained and reloaded successfully", "metrics": new_metrics}
    except Exception as e:
        logger.error(f"Retraining failure: {str(e)}")
        raise APIError(f"Model retraining failed: {str(e)}")
