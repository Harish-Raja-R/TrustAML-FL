import os
import subprocess

def setup():
    print("Setting up TrustAML-FL...")
    
    # 1. Check directories
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 2. Check dependencies (basic check)
    try:
        import torch
        print(f"PyTorch found: {torch.__version__}")
        try:
            import torch_geometric
            print(f"PyTorch Geometric found: {torch_geometric.__version__}")
        except ImportError:
            print("WARNING: PyTorch Geometric NOT found. GNN baselines will be skipped.")
    except ImportError:
        print("ERROR: PyTorch NOT found. Please install requirements.txt")
        
    print("\nSetup complete. You can now run the pipeline:")
    print("python experiments/run_full_pipeline.py")

if __name__ == "__main__":
    setup()
