from fastapi import APIRouter
import json
import os

router = APIRouter()

@router.get("/")
def list_experiments():
    results = {}
    metrics_dir = "results/metrics"
    if os.path.exists(metrics_dir):
        for f in os.listdir(metrics_dir):
            if f.endswith(".json"):
                with open(os.path.join(metrics_dir, f), "r") as file:
                    results[f.split(".")[0]] = json.load(file)
    return results
