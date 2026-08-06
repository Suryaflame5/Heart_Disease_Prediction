# Heart Disease Prediction - ECG Monitor

A production-grade standalone machine learning project built with React, TypeScript, FastAPI, and Scikit-learn, styled using Tailwind CSS and Framer Motion under the **ECG Monitor** visual system.

## Setup & Running

### Using Docker (Recommended)
1. Build and launch containers:
   ```bash
   docker-compose up --build
   ```
2. Open the frontend: [http://localhost:3000](http://localhost:3000)
3. Open API docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

### Local Manual Run
1. **Backend setup**:
   ```bash
   pip install -r requirements.txt
   python datasets/generate_data.py
   python models/train.py
   PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
   ```
2. **Frontend setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
