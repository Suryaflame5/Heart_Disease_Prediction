<div align="center">

# Heart Disease Prediction

**A production-grade Machine Learning web application**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3-38BDF8?logo=tailwindcss&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

**Heart Disease Prediction** predicts **Target** using a trained **GradientBoostingClassifier** model.

The application is built as a fully decoupled microservice:
- A **FastAPI** backend exposes REST endpoints for single prediction, batch prediction, and history retrieval.
- A **React + Vite + TypeScript** frontend provides a polished dark-mode UI with real-time prediction feedback, animated charts, and CSV batch upload.
- An **SQLite** database logs every prediction with timestamps and latency.
- A **Dockerised** setup lets anyone clone and run the full stack in one command.

---

## Features

- **Single Prediction** — Enter feature values and get an instant ML prediction with confidence scores
- **Batch Prediction** — Upload a CSV file and download results with predictions appended
- **Prediction History** — All predictions are logged to SQLite and displayed in a live table
- **Model Metrics** — View training accuracy, R² / F1 score and feature importances on the dashboard
- **Interactive Charts** — Recharts-powered visualisations update with each prediction
- **Dark Mode UI** — Glassmorphism design with Framer Motion animations
- **Swagger API Docs** — Full OpenAPI documentation at `/docs`
- **Docker Support** — One-command `docker compose up --build` deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, TailwindCSS, Framer Motion |
| Charts | Recharts |
| Backend | FastAPI, Python 3.10, Uvicorn |
| ML | Scikit-learn, GradientBoostingClassifier |
| Database | SQLite (via Python `sqlite3`) |
| Model Storage | Joblib |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
Heart_Disease_Prediction/
├── backend/
│   ├── main.py          # FastAPI app, routes, middleware
│   └── schemas.py       # Pydantic request/response models
├── datasets/
│   └── generate_data.py # Synthetic dataset generator
├── frontend/
│   └── src/
│       └── App.tsx      # Full React SPA
├── models/
│   └── train.py         # Model training + serialisation
├── saved_models/
│   └── model.joblib     # Serialised trained model (git-ignored)
├── tests/
│   └── test_api.py      # Pytest backend test suite
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1 — Docker (Recommended)

> **Requires**: Docker Desktop installed and running

```bash
git clone https://github.com/Suryaflame5/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

---

### Option 2 — Local Development

#### Backend

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate dataset and train model
python datasets/generate_data.py
python models/train.py

# Start backend
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/metrics` | Model accuracy metrics |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict-batch` | Batch CSV prediction |
| `GET` | `/history` | Prediction history logs |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 55, "Sex": "Male", "ChestPainType": "Typical Angina", "MaxHeartRate": 140, "RestBP": 120, "Cholesterol": 220, "ST_Depression": 1.2}'
```

Response:
```json
{
  "success": true,
  "prediction": {
    "val": "predicted_value",
    "probabilities": null
  },
  "latency_ms": 2.1
}
```

---

## Dataset

The dataset is **synthetically generated** via `datasets/generate_data.py` using realistic statistical distributions calibrated to the domain.

| Split | Rows |
|-------|------|
| Training | 1,600 |
| Test | 400 |

To regenerate:
```bash
python datasets/generate_data.py
python models/train.py
```

---

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   React Frontend    │  HTTP   │    FastAPI Backend    │
│  (Vite + Tailwind)  │ ──────► │  (Uvicorn, port 8000)│
│    port 3000        │ ◄────── │                      │
└─────────────────────┘  JSON   └───────────┬──────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                         ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
                         │ Sklearn │   │ SQLite  │   │ Joblib  │
                         │  Model  │   │  Logs   │   │  Store  │
                         └─────────┘   └─────────┘   └─────────┘
```

---

## CI / CD

Every push to `main` triggers the GitHub Actions pipeline (`.github/workflows/ci.yml`):

1. Install Python + backend dependencies
2. Run `pytest tests/`
3. Verify FastAPI app imports cleanly
4. Install Node + frontend dependencies
5. Run `npm run build`

---

## Future Improvements

- [ ] Add user authentication (JWT)
- [ ] Deploy to cloud (Railway / Render / GCP Cloud Run)
- [ ] Real-world dataset integration
- [ ] Add SHAP explainability charts
- [ ] Model versioning and A/B testing
- [ ] Prometheus + Grafana monitoring

---

## License

MIT © Suryaflame5
