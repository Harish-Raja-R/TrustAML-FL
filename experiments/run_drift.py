import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.drift.detectors import detect_feature_drift

def run_drift():
    print("Loading data for Temporal Drift Detection...")
    loader = DataLoader()
    tx_df, _ = loader.load_or_generate_data(num_banks=3, num_transactions=10000)
    
    # Sort chronologically
    tx_df = tx_df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Split into two windows
    mid = len(tx_df) // 2
    window_1 = tx_df.iloc[:mid]
    window_2 = tx_df.iloc[mid:]
    
    # We pretend window_2 has a massive shift in amount
    # (Since we generated randomly, there might not be real drift, so we artificially inject it for the experiment)
    window_2_drifted = window_2.copy()
    window_2_drifted['amount'] = window_2_drifted['amount'] * 10
    
    features_1 = window_1[['amount']].values
    features_2 = window_2_drifted[['amount']].values
    
    drifts = detect_feature_drift(features_1, features_2)
    print(f"Detected Drift between Window 1 and Window 2: {drifts}")
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/drift.json", "w") as f:
        json.dump(drifts, f, indent=4)
    print("\nDrift experiment completed. Results saved to results/metrics/drift.json")

if __name__ == "__main__":
    run_drift()
