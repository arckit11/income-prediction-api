from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictionInput(BaseModel):
    age: int = Field(..., ge=17, le=90, description="Age in years", example=35)
    workclass: str = Field(..., description="Employment type", example="Private")
    fnlwgt: int = Field(..., ge=0, description="Final weight (census sampling weight)", example=215646)
    education: str = Field(..., description="Highest education level", example="Bachelors")
    education_num: int = Field(..., ge=1, le=16, description="Education level as number", example=13)
    marital_status: str = Field(..., description="Marital status", example="Never-married")
    occupation: str = Field(..., description="Job type", example="Tech-support")
    relationship: str = Field(..., description="Relationship in household", example="Not-in-family")
    race: str = Field(..., description="Race", example="White")
    sex: Literal["Male", "Female"] = Field(..., description="Sex", example="Male")
    capital_gain: float = Field(..., ge=0, description="Capital gains", example=0)
    capital_loss: float = Field(..., ge=0, description="Capital losses", example=0)
    hours_per_week: int = Field(..., ge=1, le=99, description="Hours worked per week", example=40)
    native_country: str = Field(..., description="Country of origin", example="United-States")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlwgt": 215646,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }


class PredictionOutput(BaseModel):
    income_class: Literal["<=50K", ">50K"]
    probability_high_income: float
    confidence: Literal["High", "Medium", "Low"]
    confidence_note: str


class BatchPredictionOutput(BaseModel):
    total_records: int
    predictions: list[dict]
    summary: dict


class StatsOutput(BaseModel):
    total_predictions: int
    high_income_count: int
    low_income_count: int
    high_income_pct: float


class ValidValuesOutput(BaseModel):
    workclass: list[str]
    education: list[str]
    marital_status: list[str]
    occupation: list[str]
    relationship: list[str]
    race: list[str]
    sex: list[str]
    native_country: list[str]
