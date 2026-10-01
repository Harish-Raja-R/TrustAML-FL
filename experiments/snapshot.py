import os
import sys
import json
import subprocess
import torch
try:
    import torch_geometric
except ImportError:
    torch_geometric = None
import sklearn
import datetime

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def create_snapshot():
    os.makedirs("results", exist_ok=True)
    
    env_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_revision_hash(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__ if torch_geometric else "not_installed",
        "sklearn_version": sklearn.__version__,
        "cpu_info": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    with open("results/phase12_environment.json", "w") as f:
        json.dump(env_info, f, indent=4)
        
    baseline_snapshot = {
        "status": "pre_phase12",
        "description": "Baseline before Phase 12 modifications.",
        "environment": env_info
    }
    
    with open("results/phase12_baseline_snapshot.json", "w") as f:
        json.dump(baseline_snapshot, f, indent=4)
        
    print("Snapshot created successfully.")

if __name__ == "__main__":
    create_snapshot()
