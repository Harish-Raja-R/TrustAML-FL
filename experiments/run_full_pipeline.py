import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all():
    print("=== TrustAML-FL Full Pipeline ===")
    
    # Normally we would call functions directly. Since we have standalone scripts, 
    # we can import and run their main logic.
    
    # Due to imports above, we can just do system calls or refactor the scripts.
    # Refactoring the scripts to have a main function is better, but since we didn't 
    # refactor all of them perfectly (e.g. run_centralized does logic in __main__),
    # let's just use os.system for simplicity in the prototype.
    
    scripts = [
        "experiments/run_centralized.py",
        "experiments/run_local.py",
        "experiments/run_fedavg.py",
        "experiments/run_fedprox.py",
        "experiments/run_privacy.py",
        "experiments/run_robustness.py",
        "experiments/run_drift.py",
        "experiments/generate_tables.py"
    ]
    
    for script in scripts:
        print(f"\n>>> Running {script} <<<")
        os.system(f"python {script}")
        
    print("\n=== Full Pipeline Completed ===")

if __name__ == "__main__":
    run_all()
