from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ml.linear_regression import get_linear_regression_info, predict_exam_score


app = FastAPI(title="Interactive ML Learning Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LinearRegressionPredictionRequest(BaseModel):
    hours_studied: float = Field(
        ...,
        ge=0,
        le=24,
        description="Student study hours"
    )

#===========================================================

@app.get("/")
def root():
    return {
        "project": "Interactive ML Learning Dashboard",
        "description": "Turkish interactive dashboard for learning classical Machine Learning algorithms",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "message": "Interactive ML Learning Dashboard API is running"
    }


@app.get("/linear-regression")
def linear_regression_info():
    return get_linear_regression_info()


@app.post("/linear-regression/predict")
def linear_regression_predict(request: LinearRegressionPredictionRequest):
    return predict_exam_score(request.hours_studied)