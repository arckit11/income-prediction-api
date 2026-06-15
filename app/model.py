import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException

MODEL_DIR = Path(__file__).parent.parent / "model"

# Load all artifacts once at startup
_model            = joblib.load(MODEL_DIR / "xgb_model.pkl")
_encoders         = joblib.load(MODEL_DIR / "encoders.pkl")
_le_income        = joblib.load(MODEL_DIR / "le_income.pkl")
_scaler           = joblib.load(MODEL_DIR / "scaler.pkl")
_power_transformer= joblib.load(MODEL_DIR / "power_transformer.pkl")
_feature_names    = joblib.load(MODEL_DIR / "feature_names.pkl")
_label_maps       = joblib.load(MODEL_DIR / "label_maps.pkl")

# Exact column order from training (matches notebook Cell 71)
FEATURE_ORDER = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship', 'race', 'sex',
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country'
]

# Map from API field names (underscores) to CSV column names (dots)
API_TO_CSV = {
    'education_num':   'education.num',
    'marital_status':  'marital.status',
    'capital_gain':    'capital.gain',
    'capital_loss':    'capital.loss',
    'hours_per_week':  'hours.per.week',
    'native_country':  'native.country',
}

CAT_COLS = ['workclass','education','occupation','marital.status',
            'relationship','race','sex','native.country']

def get_valid_values() -> dict:
    """Return all valid string values for each categorical field."""
    return {col: list(enc.classes_) for col, enc in _encoders.items()}

def _encode_row(data: dict) -> pd.DataFrame:
    """
    Apply identical preprocessing to a single row as the notebook did:
    1. Rename API fields to CSV column names
    2. Label-encode categoricals (with unknown-value check)
    3. Remove outliers: NOT applied (single row — no distribution)
    4. Power-transform age, fnlwgt
    5. Standard-scale all features
    """
    # 1. Rename
    renamed = {}
    for k, v in data.items():
        csv_key = API_TO_CSV.get(k, k)
        renamed[csv_key] = v

    # 2. Label-encode categoricals
    for col in CAT_COLS:
        val = str(renamed[col])
        if val not in _encoders[col].classes_:
            valid = list(_encoders[col].classes_)
            raise HTTPException(
                status_code=422,
                detail=f"Invalid value '{val}' for field '{col}'. Valid: {valid}"
            )
        renamed[col] = int(_encoders[col].transform([val])[0])

    # 3. Build DataFrame in exact feature order
    df = pd.DataFrame([renamed])[FEATURE_ORDER]

    # 4. Power-transform age and fnlwgt (must be positive; already is)
    df[['age', 'fnlwgt']] = _power_transformer.transform(df[['age', 'fnlwgt']])

    # 5. Standard-scale
    df_scaled = pd.DataFrame(
        _scaler.transform(df),
        columns=FEATURE_ORDER
    )
    return df_scaled


def predict_single(data: dict) -> dict:
    df = _encode_row(data)
    prob = float(_model.predict_proba(df)[0][1])
    label_encoded = int(_model.predict(df)[0])
    label = str(_le_income.inverse_transform([label_encoded])[0])

    if prob >= 0.80 or prob <= 0.20:
        confidence = "High"
        note = "Model is highly confident in this prediction."
    elif prob >= 0.65 or prob <= 0.35:
        confidence = "Medium"
        note = "Moderate confidence. Consider borderline cases carefully."
    else:
        confidence = "Low"
        note = "Low confidence — result is near decision boundary (50%)."

    return {
        "income_class":            label,
        "probability_high_income": round(prob, 4),
        "confidence":              confidence,
        "confidence_note":         note,
    }


def predict_batch(df_raw: pd.DataFrame) -> list[dict]:
    """Process a CSV dataframe — expects original column names from adult.csv."""
    results = []
    for idx, row in df_raw.iterrows():
        try:
            # map dot-column names back to API underscore names for reuse
            api_row = {}
            for feature in FEATURE_ORDER:
                api_key = {v: k for k, v in API_TO_CSV.items()}.get(feature, feature)
                api_row[api_key] = row[feature]
            result = predict_single(api_row)
            result['row_index'] = int(idx)
            results.append(result)
        except Exception as e:
            results.append({"row_index": int(idx), "error": str(e)})
    return results
