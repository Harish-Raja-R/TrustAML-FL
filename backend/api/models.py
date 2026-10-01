from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_models():
    return [
        {"id": "global_fedavg", "type": "HeteroGraphSAGE", "status": "active"},
        {"id": "global_fedprox", "type": "HeteroGraphSAGE", "status": "available"}
    ]
