import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.drift.detectors import detect_feature_drift

def run_synthetic_drift(tx_df):
    print("\n--- Running Synthetic Injected Drift ---")
    mid = len(tx_df) // 2
    window_1 = tx_df.iloc[:mid]
    window_2 = tx_df.iloc[mid:].copy()
    window_2['amount'] = window_2['amount'] * 10
    
    features_1 = window_1[['amount']].values
    features_2 = window_2[['amount']].values
    
    drifts = detect_feature_drift(features_1, features_2)
    print(f"Synthetic Injected Drift: {drifts}")
    return drifts

def run_natural_drift(tx_df):
    print("\n--- Running Natural Temporal Windows Drift ---")
    windows = np.array_split(tx_df, 5)
    
    results = {}
    base_window = windows[0]
    base_features = base_window[['amount']].values
    
    for i in range(1, 5):
        target_window = windows[i]
        target_features = target_window[['amount']].values
        drifts = detect_feature_drift(base_features, target_features)
        
        # AML Class rate shift
        base_aml_rate = base_window['is_aml'].mean()
        target_aml_rate = target_window['is_aml'].mean()
        
        drifts['aml_rate_shift'] = float(target_aml_rate - base_aml_rate)
        results[f"Window_{i+1}"] = drifts
        print(f"Window {i+1} vs 1: {drifts}")
        
    return results

def run_drift():
    print("Loading data for Temporal Drift Detection...")
    loader = DataLoader()
    tx_df, _ = loader.load_or_generate_data(num_banks=3, num_transactions=10000)
    
    # Sort chronologically
    tx_df = tx_df.sort_values(by="timestamp").reset_index(drop=True)
    
    synthetic_results = run_synthetic_drift(tx_df)
    natural_results = run_natural_drift(tx_df)
    
    all_results = {
        "synthetic_injected_drift": synthetic_results,
        "natural_temporal_windows": natural_results
    }
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/drift.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nDrift experiment completed. Results saved to results/metrics/drift.json")

if __name__ == "__main__":
    run_drift()
