from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
import io

from app.schema import (
    PredictionInput, PredictionOutput,
    BatchPredictionOutput, StatsOutput, ValidValuesOutput
)
from app.model import predict_single, predict_batch, get_valid_values
from app.logger import init_db, log_prediction, get_stats

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(
    title="Income Prediction API",
    description=(
        "Predicts whether an individual's income exceeds $50K/year using US Census data.\n\n"
        "Model: XGBoost (tuned with GridSearchCV)  |  Accuracy: 84.29%  |  Dataset: UCI Adult Census (48K records)\n\n"
        "Built by Arckit Arihant — github.com/arckit11"
    ),
    version="1.0.0",
    contact={"name": "Arckit Arihant", "url": "https://github.com/arckit11"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":   "running",
        "model":    "XGBoost (GridSearchCV tuned)",
        "accuracy": "84.29%",
        "dataset":  "UCI Adult Census (48,842 records)",
        "endpoints": {
            "/predict":       "POST - single prediction",
            "/batch-predict": "POST - CSV batch prediction",
            "/stats":         "GET  - prediction log stats",
            "/valid-values":  "GET  - all valid input values",
            "/docs":          "GET  - interactive Swagger UI",
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Predict income class for a single individual.

    Returns:
    - **income_class**: `<=50K` or `>50K`
    - **probability_high_income**: probability score (0-1) for >50K class
    - **confidence**: High / Medium / Low based on probability distance from 0.5
    - **confidence_note**: plain-English explanation of confidence
    """
    data = input_data.model_dump()
    result = predict_single(data)
    log_prediction(data, result)
    return result


@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["Prediction"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict income class for multiple records from a CSV file.

    CSV must have these columns (same as adult.csv):
    age, workclass, fnlwgt, education, education.num, marital.status,
    occupation, relationship, race, sex, capital.gain, capital.loss,
    hours.per.week, native.country
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required_cols = [
        'age','workclass','fnlwgt','education','education.num','marital.status',
        'occupation','relationship','race','sex','capital.gain','capital.loss',
        'hours.per.week','native.country'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    results = predict_batch(df)
    high = sum(1 for r in results if r.get("income_class") == ">50K")
    low  = sum(1 for r in results if r.get("income_class") == "<=50K")

    return {
        "total_records": len(results),
        "predictions":   results,
        "summary": {
            "high_income_count": high,
            "low_income_count":  low,
            "high_income_pct":   round(high / len(results) * 100, 1) if results else 0,
        }
    }


@app.get("/stats", response_model=StatsOutput, tags=["Monitoring"])
def stats():
    """
    Return aggregate statistics of all predictions made since the API started.
    Useful for monitoring prediction distribution and detecting data drift.
    """
    return get_stats()


@app.get("/valid-values", response_model=ValidValuesOutput, tags=["Reference"])
def valid_values():
    """
    Return all valid string values for each categorical input field.
    Use this to validate inputs before calling /predict.
    """
    vals = get_valid_values()
    return {
        "workclass":      vals["workclass"].tolist() if hasattr(vals["workclass"], "tolist") else list(vals["workclass"]),
        "education":      list(vals["education"]),
        "marital_status": list(vals["marital.status"]),
        "occupation":     list(vals["occupation"]),
        "relationship":   list(vals["relationship"]),
        "race":           list(vals["race"]),
        "sex":            list(vals["sex"]),
        "native_country": list(vals["native.country"]),
    }
