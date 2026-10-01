from fastapi import APIRouter
import json
import os

router = APIRouter()

def read_json_safe(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception:
        return None

@router.get("/")
def list_experiments():
    metrics_dir = "results/metrics"
    
    return {
        "centralized": read_json_safe(os.path.join(metrics_dir, "centralized.json")),
        "local": read_json_safe(os.path.join(metrics_dir, "local.json")),
        "fedavg": read_json_safe(os.path.join(metrics_dir, "fedavg.json")),
        "fedprox": read_json_safe(os.path.join(metrics_dir, "fedprox.json")),
        "privacy": read_json_safe(os.path.join(metrics_dir, "privacy.json")),
        "robustness": read_json_safe(os.path.join(metrics_dir, "robustness.json")),
        "drift": read_json_safe(os.path.join(metrics_dir, "drift.json"))
    }
