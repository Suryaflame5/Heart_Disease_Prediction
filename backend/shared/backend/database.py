import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class PredictionHistoryDB:
    def __init__(self, db_path: str = "history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                )
            """)
            conn.commit()

    def log_prediction(self, project_name: str, input_data: Dict[str, Any], prediction: Any, latency_ms: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO prediction_history (timestamp, project_name, input_data, prediction, latency_ms) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.utcnow().isoformat() + "Z",
                    project_name,
                    json.dumps(input_data),
                    json.dumps(prediction),
                    latency_ms
                )
            )
            conn.commit()

    def get_history(self, project_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, timestamp, input_data, prediction, latency_ms FROM prediction_history WHERE project_name = ? ORDER BY id DESC LIMIT ?",
                (project_name, limit)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "input_data": json.loads(row["input_data"]),
                    "prediction": json.loads(row["prediction"]),
                    "latency_ms": row["latency_ms"]
                }
                for row in rows
            ]

    def clear_history(self, project_name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM prediction_history WHERE project_name = ?", (project_name,))
            conn.commit()
