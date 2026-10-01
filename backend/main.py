from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import system, experiments, alerts, models

app = FastAPI(title="TrustAML-FL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["System"])
# app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
# app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
# app.include_router(models.router, prefix="/api/models", tags=["Models"])

@app.get("/")
def root():
    return {"message": "TrustAML-FL Backend is running."}
