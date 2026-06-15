# Income Prediction API

Production REST API predicting whether an individual earns >$50K/year from US Census demographic data.

**Model:** XGBoost (GridSearchCV tuned) | **Accuracy:** 84.29% | **Dataset:** UCI Adult Census (48,842 records)

> **Live API:** `https://income-prediction-api.onrender.com` ← add your Render URL here after deploying
>
> **Interactive docs:** `https://income-prediction-api.onrender.com/docs`

---

## Architecture

```
POST /predict
     │
     ▼
Pydantic validation  ──► 422 error if invalid input
     │
     ▼
Label Encoding (8 categorical columns)
     │
     ▼
Power Transform (Box-Cox on age, fnlwgt)
     │
     ▼
Standard Scaling (all 14 features)
     │
     ▼
XGBoost model.predict_proba()
     │
     ▼
Confidence scoring + SQLite logging
     │
     ▼
JSON response {income_class, probability, confidence}
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Browser dashboard UI |
| `GET`  | `/health` | Health check + endpoint index |
| `POST` | `/predict` | Single prediction with confidence score |
| `POST` | `/batch-predict` | Upload CSV for bulk predictions |
| `GET`  | `/stats` | Prediction log statistics (monitoring) |
| `GET`  | `/valid-values` | All valid categorical input values |
| `GET`  | `/docs` | Interactive Swagger UI |

---

## Model details

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost (tuned) |
| Best params | learning_rate=0.3, max_depth=10, n_estimators=11 |
| Accuracy | 84.29% |
| Precision (>50K) | 0.67 |
| Recall (>50K) | 0.77 |
| F1 (>50K) | 0.71 |
| Dataset | UCI Adult Census, 48,842 records |
| Class imbalance | Handled with SMOTE (strategy=0.80) |
| Preprocessing | Label encoding, IQR outlier removal, Box-Cox transform, StandardScaler |

---

## Sample request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "workclass": "Private",
    "fnlwgt": 215646,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
  }'
```

**Response:**
```json
{
  "income_class": ">50K",
  "probability_high_income": 0.7754,
  "confidence": "Medium",
  "confidence_note": "Moderate confidence. Consider borderline cases carefully."
}
```

---

## Local setup

```bash
git clone https://github.com/arckit11/income-prediction-api
cd income-prediction-api

pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Open in browser
# http://localhost:8000/      ← dashboard UI
# http://localhost:8000/docs ← Swagger UI
# http://localhost:8000/health ← health check
```

## Docker

```bash
docker build -t income-prediction-api .
docker run -p 8000:8000 income-prediction-api
```

## Deploy to Render

1. Push this repository to GitHub.
2. Create a new Web Service on Render.
3. Select **Docker** as the environment.
4. Connect the repository and choose the branch.
5. Render will use the project `Dockerfile` and start the service on port `8000`.
6. Once deployed, visit `https://<your-render-service>.onrender.com/` for the dashboard and `https://<your-render-service>.onrender.com/docs` for API docs.

---

## Deploy to Render (free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Runtime: **Docker**
5. Instance type: **Free**
6. Click **Deploy**

Your API will be live at `https://income-prediction-api.onrender.com` in ~3 minutes.

---

## Project structure

```
income-prediction-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, all endpoints
│   ├── model.py         # Preprocessing pipeline + XGBoost inference
│   ├── schema.py        # Pydantic input/output validation
│   └── logger.py        # SQLite prediction logging
├── model/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── encoders.pkl           # LabelEncoders for 8 categorical columns
│   ├── scaler.pkl             # StandardScaler
│   ├── power_transformer.pkl  # Box-Cox transformer (age, fnlwgt)
│   ├── feature_names.pkl      # Ordered feature list
│   ├── label_maps.pkl         # Human-readable category mappings
│   └── le_income.pkl          # Income label encoder
├── requirements.txt
├── Dockerfile
└── README.md
```

---

Built by [Arckit Arihant](https://github.com/arckit11) · B.E. CSE (AI/ML), RNSIT Bengaluru
