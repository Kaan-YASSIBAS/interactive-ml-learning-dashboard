from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Interactive ML Learning Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "message": "Interactive ML Learning Dashboard API is running"
    }


@app.get("/")
def root():
    return {
        "project": "Interactive ML Learning Dashboard",
        "description": "Turkish interactive dashboard for learning classical Machine Learning algorithms",
        "status": "running"
    }