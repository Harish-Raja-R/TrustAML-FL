from fastapi import APIRouter
import psutil

router = APIRouter()

@router.get("/status")
def get_system_status():
    return {
        "status": "online",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }
