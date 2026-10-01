"""
Phase 12: Generate all publication-ready tables from existing experiment results.
Reads from results/metrics/*.json, results/oot_results.csv, results/noniid_results_raw.csv, etc.
Does NOT fabricate any values — only reformats and aggregates existing data.
"""
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("results/tables", exist_ok=True)

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def save_table(df, name):
    csv_path = f"results/tables/{name}.csv"
    md_path = f"results/tables/{name}.md"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"  Saved {csv_path}")

# ============================================================
# Table 1: Dataset and Graph Statistics
# ============================================================
def table1_dataset_stats():
    print("\n=== Table 1: Dataset and Graph Statistics ===")
    tx_path = "data/synthetic/transactions.csv"
    acc_path = "data/synthetic/accounts.csv"
    if not os.path.exists(tx_path):
        print("  [SKIP] No transactions.csv found.")
        return
    tx = pd.read_csv(tx_path)
    acc = pd.read_csv(acc_path)
    
    rows = [
        {"Statistic": "Total Transactions", "Value": len(tx)},
        {"Statistic": "Total Accounts", "Value": len(acc)},
        {"Statistic": "Unique Devices", "Value": tx['device'].nunique()},
        {"Statistic": "Unique Locations", "Value": tx['location'].nunique()},
        {"Statistic": "AML Transactions", "Value": int(tx['is_aml'].sum())},
        {"Statistic": "AML Rate (%)", "Value": round(tx['is_aml'].mean() * 100, 2)},
        {"Statistic": "Number of Banks", "Value": tx['sender_bank'].nunique() if 'sender_bank' in tx.columns else "N/A"},
        {"Statistic": "Date Range", "Value": f"{tx['timestamp'].min()} to {tx['timestamp'].max()}"},
        {"Statistic": "Mean Amount", "Value": round(tx['amount'].mean(), 2)},
        {"Statistic": "Median Amount", "Value": round(tx['amount'].median(), 2)},
    ]
    df = pd.DataFrame(rows)
    save_table(df, "Table_1_Dataset_Statistics")
    print(df)

# ============================================================
# Table 2: Centralized vs Local vs FL
# ============================================================
def table2_baseline():
    print("\n=== Table 2: Centralized vs Local vs FL ===")
    methods = {
        "Centralized": "results/metrics/centralized.json",
        "FedAvg": "results/metrics/fedavg.json",
        "FedProx": "results/metrics/fedprox.json",
        "Local": "results/metrics/local.json",
    }
    rows = []
    for method, path in methods.items():
        data = load_json(path)
        if data is None:
            continue
        # Get the last entry (final round or final result)
        if isinstance(data, list):
            m = data[-1]
        elif isinstance(data, dict):
            # Might be nested; try to get last value
            if "metrics" in data:
                m = data["metrics"][-1] if isinstance(data["metrics"], list) else data["metrics"]
            else:
                m = data
        else:
            continue
        rows.append({
            "Method": method,
            "PR-AUC": round(m.get("pr_auc", 0), 4),
            "ROC-AUC": round(m.get("roc_auc", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "MCC": round(m.get("mcc", 0), 4),
        })
    if rows:
        df = pd.DataFrame(rows)
        save_table(df, "Table_2_Baseline_Comparison")
        print(df)

# ============================================================
# Table 3: IID vs Non-IID (from multi-seed run)
# ============================================================
def table3_noniid():
    print("\n=== Table 3: IID vs Non-IID ===")
    path = "results/noniid_results_raw.csv"
    if not os.path.exists(path):
        # Try tables path
        path = "results/tables/Table_3_NonIID.csv"
    if not os.path.exists(path):
        print("  [SKIP] No noniid results found.")
        return
    df = pd.read_csv(path)
    print(df)
    # Already saved by run_noniid_seeds.py; copy to ensure consistency
    save_table(df, "Table_3_NonIID_Results")

# ============================================================
# Table 5: Privacy–Utility Tradeoff
# ============================================================
def table5_privacy():
    print("\n=== Table 5: Privacy-Utility Tradeoff ===")
    data = load_json("results/metrics/privacy.json")
    if data is None:
        return
    rows = []
    for dp_name, dp_data in data.items():
        metrics_list = dp_data.get("metrics", [])
        final = metrics_list[-1] if isinstance(metrics_list, list) and metrics_list else {}
        privacy = dp_data.get("privacy_status", {})
        rows.append({
            "DP_Level": dp_name,
            "PR-AUC": round(final.get("pr_auc", 0), 4),
            "ROC-AUC": round(final.get("roc_auc", 0), 4),
            "F1": round(final.get("f1", 0), 4),
            "Noise_Multiplier": privacy.get("noise_multiplier", "N/A"),
            "Epsilon": privacy.get("epsilon", "not_computed"),
            "Status": privacy.get("status", "unknown"),
        })
    df = pd.DataFrame(rows)
    save_table(df, "Table_5_Privacy_Utility")
    print(df)

# ============================================================
# Table 6: Robustness
# ============================================================
def table6_robustness():
    print("\n=== Table 6: Robustness ===")
    data = load_json("results/metrics/robustness.json")
    if data is None:
        return
    rows = []
    for exp_name, exp_data in data.items():
        rows.append({
            "Experiment": exp_name,
            "Aggregation": exp_data.get("aggregation", ""),
            "Malicious_%": exp_data.get("malicious_percent", 0),
            "PR-AUC": round(exp_data.get("final_pr_auc", 0), 4),
            "F1": round(exp_data.get("final_f1", 0), 4),
        })
    df = pd.DataFrame(rows)
    save_table(df, "Table_6_Robustness")
    print(df)

# ============================================================
# Table 7: Temporal Drift
# ============================================================
def table7_drift():
    print("\n=== Table 7: Temporal Drift ===")
    data = load_json("results/metrics/drift.json")
    if data is None:
        return
    rows = []
    # Natural temporal windows
    nat = data.get("natural_temporal_windows", {})
    for window, wdata in nat.items():
        rows.append({
            "Type": "Natural",
            "Window": window,
            "AML_Rate_Shift": round(wdata.get("aml_rate_shift", 0), 4),
        })
    # Synthetic injected
    syn = data.get("synthetic_injected_drift", {})
    if syn:
        rows.append({
            "Type": "Synthetic_Injected",
            "Window": "injected",
            "AML_Rate_Shift": round(syn.get("aml_rate_shift", 0), 4),
        })
    if rows:
        df = pd.DataFrame(rows)
        save_table(df, "Table_7_Temporal_Drift")
        print(df)

# ============================================================
# Table 8: Static vs Drift-Aware
# ============================================================
def table8_adaptation():
    print("\n=== Table 8: Static vs Drift-Aware ===")
    path = "results/tables/Table_8_Drift_Adaptation.csv"
    if not os.path.exists(path):
        print("  [SKIP] No drift adaptation results found.")
        return
    df = pd.read_csv(path)
    # Re-save to ensure consistent naming
    save_table(df, "Table_8_Drift_Adaptation")
    print(df)

# ============================================================
# Table 9: OOT Results
# ============================================================
def table9_oot():
    print("\n=== Table 9: Out-of-Time Results ===")
    path = "results/oot_results.csv"
    if not os.path.exists(path):
        print("  [SKIP] No OOT results found.")
        return
    df = pd.read_csv(path)
    save_table(df, "Table_9_OOT_Results")
    print(df)

# ============================================================
# Table 10: Explainability Cases
# ============================================================
def table10_explainability():
    print("\n=== Table 10: Explainability Cases ===")
    data = load_json("results/explainability_cases.json")
    if data is None:
        return
    rows = []
    for case_type, case_data in data.items():
        rows.append({
            "Case": case_type,
            "Transaction_ID": case_data.get("transaction_id", ""),
            "Risk_Score": round(case_data.get("risk_score", 0), 4),
            "Risk_Level": case_data.get("risk_level", ""),
        })
    df = pd.DataFrame(rows)
    save_table(df, "Table_10_Explainability")
    print(df)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 12: Generating Publication-Ready Tables")
    print("=" * 60)
    table1_dataset_stats()
    table2_baseline()
    table3_noniid()
    table5_privacy()
    table6_robustness()
    table7_drift()
    table8_adaptation()
    table9_oot()
    table10_explainability()
    print("\n" + "=" * 60)
    print("Table generation complete.")
    print("=" * 60)
