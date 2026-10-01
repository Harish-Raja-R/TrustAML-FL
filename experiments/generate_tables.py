import os
import json
import pandas as pd

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_tables():
    print("Generating Research Tables...")
    metrics_dir = "results/metrics"
    
    rob = load_json(os.path.join(metrics_dir, "robustness.json"))
    if rob:
        print("\n=== Table: Robustness ===")
        rows = []
        for exp, data in rob.items():
            rows.append({
                "Experiment": exp,
                "Aggregation": data.get("aggregation"),
                "Malicious_Pct": data.get("malicious_percent"),
                "PR-AUC": data.get("final_pr_auc"),
                "F1": data.get("final_f1")
            })
        df_rob = pd.DataFrame(rows)
        print(df_rob)
        df_rob.to_csv("results/metrics/table_robustness.csv", index=False)
        
    priv = load_json(os.path.join(metrics_dir, "privacy.json"))
    if priv:
        print("\n=== Table: Privacy ===")
        rows = []
        for exp, data in priv.items():
            final_metrics = data.get("metrics", [])[-1] if data.get("metrics") else {}
            acc = data.get("privacy_status", {})
            rows.append({
                "Setting": exp,
                "PR-AUC": final_metrics.get("pr_auc", 0),
                "ROC-AUC": final_metrics.get("roc_auc", 0),
                "Status": acc.get("status")
            })
        df_priv = pd.DataFrame(rows)
        print(df_priv)
        df_priv.to_csv("results/metrics/table_privacy.csv", index=False)
        
    drift = load_json(os.path.join(metrics_dir, "drift.json"))
    if drift:
        print("\n=== Table: Temporal Drift ===")
        rows = []
        nat = drift.get("natural_temporal_windows", {})
        for window, metrics in nat.items():
            rows.append({
                "Window": window,
                "Wasserstein": metrics.get("feature_0", {}).get("wasserstein", 0),
                "KL_Div": metrics.get("feature_0", {}).get("kl", 0),
                "AML_Rate_Shift": metrics.get("aml_rate_shift", 0)
            })
        df_drift = pd.DataFrame(rows)
        print(df_drift)
        df_drift.to_csv("results/metrics/table_drift.csv", index=False)
        
    print("\nTables saved to results/metrics/")

if __name__ == "__main__":
    generate_tables()
