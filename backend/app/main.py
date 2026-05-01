from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ml.linear_regression import get_linear_regression_info, predict_exam_score
from app.ml.logistic_regression import get_logistic_regression_info, predict_message_class


app = FastAPI(title="Interactive ML Learning Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#=============================== Pydantic Models ===============================
class LinearRegressionPredictionRequest(BaseModel):
    hours_studied: float = Field(
        ...,
        ge=0,
        le=24,
        description="Student study hours"
    )

class LogisticRegressionPredictionRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Message text to classify as spam or ham"
    )


#=============================== API Endpoints ===============================

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

#================ Linear Regression Endpoints ================
@app.get("/linear-regression")
def linear_regression_info():
    return get_linear_regression_info()


@app.post("/linear-regression/predict")
def linear_regression_predict(request: LinearRegressionPredictionRequest):
    return predict_exam_score(request.hours_studied)

#================ Logistic Regression Endpoints ================ 
@app.get("/logistic-regression")
def logistic_regression_info():
    return get_logistic_regression_info()


@app.post("/logistic-regression/predict")
def logistic_regression_predict(request: LogisticRegressionPredictionRequest):
    return predict_message_class(request.message)