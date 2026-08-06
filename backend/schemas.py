from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionInput(BaseModel):
    Age: int = 55
    Sex: str = "Male"
    ChestPainType: str = "Typical Angina"
    MaxHeartRate: int = 140
    RestBP: int = 120
    Cholesterol: int = 220
    ST_Depression: float = 1.2

class PredictionResponse(BaseModel):
    success: bool
    prediction: Any
    latency_ms: float

class HistoryLog(BaseModel):
    id: int
    timestamp: str
    input_data: Dict[str, Any]
    prediction: Any
    latency_ms: float
