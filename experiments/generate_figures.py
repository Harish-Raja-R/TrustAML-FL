"""
Phase 12: Generate publication-quality figures from existing experiment results.
All data is read from results/ — no values are fabricated.
"""
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("WARNING: matplotlib not available. Figures will not be generated.")

os.makedirs("results/figures", exist_ok=True)

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_round_metrics(data):
    """Handle both list-of-dicts and dict-of-dicts (keyed by round number)."""
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Could be {"1": {...}, "2": {...}} or {"metrics": [...]}
        if "metrics" in data:
            return data["metrics"] if isinstance(data["metrics"], list) else []
        # Try numeric keys
        rounds = []
        for k, v in sorted(data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            if isinstance(v, dict) and "pr_auc" in v:
                v["round"] = int(k) if k.isdigit() else k
                rounds.append(v)
        return rounds
    return []

# ============================================================
# Figure 2: Federated Convergence
# ============================================================
def fig2_convergence():
    if not MPL_AVAILABLE:
        return
    print("  Fig 2: Convergence curves...")
    fedavg = load_json("results/metrics/fedavg.json")
    fedprox = load_json("results/metrics/fedprox.json")
    if not fedavg and not fedprox:
        print("    [SKIP] Missing data.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if fedavg:
        fa_rounds = extract_round_metrics(fedavg)
        if fa_rounds:
            rounds = [m.get("round", i+1) for i, m in enumerate(fa_rounds)]
            praucs = [m.get("pr_auc", 0) for m in fa_rounds]
            ax.plot(rounds, praucs, 'o-', label='FedAvg', color='#2196F3', linewidth=2)
    
    if fedprox:
        fp_rounds = extract_round_metrics(fedprox)
        if fp_rounds:
            rounds = [m.get("round", i+1) for i, m in enumerate(fp_rounds)]
            praucs = [m.get("pr_auc", 0) for m in fp_rounds]
            ax.plot(rounds, praucs, 's-', label='FedProx', color='#FF5722', linewidth=2)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('PR-AUC', fontsize=12)
    ax.set_title('Federated Learning Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/figures/fig2_convergence.png', dpi=150)
    plt.close(fig)
    print("    Saved fig2_convergence.png")

# ============================================================
# Figure 3: PR-AUC Comparison
# ============================================================
def fig3_prauc_comparison():
    if not MPL_AVAILABLE:
        return
    print("  Fig 3: PR-AUC comparison...")
    # Use multi-seed IID results as the most reliable baseline
    path = "results/noniid_results_raw.csv"
    if not os.path.exists(path):
        print("    [SKIP] No noniid results.")
        return
    df = pd.read_csv(path)
    iid = df[df['Degree'] == 'iid']
    if iid.empty:
        return
    
    methods_data = iid.groupby('Method')['PR-AUC'].mean().to_dict()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#FF5722']
    names = list(methods_data.keys())
    vals = list(methods_data.values())
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('PR-AUC (mean, IID)', fontsize=12)
    ax.set_title('Federated AML Detection: FedAvg vs FedProx (IID)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/figures/fig3_prauc_comparison.png', dpi=150)
    plt.close(fig)
    print("    Saved fig3_prauc_comparison.png")

# ============================================================
# Figure 4: Non-IID Performance
# ============================================================
def fig4_noniid():
    if not MPL_AVAILABLE:
        return
    print("  Fig 4: Non-IID performance...")
    path = "results/noniid_results_raw.csv"
    if not os.path.exists(path):
        print("    [SKIP] No noniid results.")
        return
    df = pd.read_csv(path)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'FedAvg': '#2196F3', 'FedProx': '#FF5722'}
    for method in df['Method'].unique():
        sub = df[df['Method'] == method].groupby('Degree')['PR-AUC'].agg(['mean', 'std']).reset_index()
        order = {'iid': 0, 'moderate': 1, 'severe': 2}
        sub['order'] = sub['Degree'].map(order)
        sub = sub.sort_values('order')
        ax.errorbar(sub['Degree'], sub['mean'], yerr=sub['std'], marker='o', capsize=5, 
                     linewidth=2, label=method, color=colors.get(method, 'gray'))
    ax.set_xlabel('Non-IID Degree', fontsize=12)
    ax.set_ylabel('PR-AUC (mean ± std)', fontsize=12)
    ax.set_title('Impact of Non-IID Heterogeneity on Federated AML Detection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/figures/fig4_noniid.png', dpi=150)
    plt.close(fig)
    print("    Saved fig4_noniid.png")

# ============================================================
# Figure 5: Privacy–Utility Tradeoff
# ============================================================
def fig5_privacy():
    if not MPL_AVAILABLE:
        return
    print("  Fig 5: Privacy-utility tradeoff...")
    data = load_json("results/metrics/privacy.json")
    if data is None:
        return
    
    names = []
    praucs = []
    for dp_name, dp_data in data.items():
        metrics_list = dp_data.get("metrics", [])
        if isinstance(metrics_list, list) and metrics_list:
            final = metrics_list[-1]
        elif isinstance(metrics_list, dict):
            vals = list(metrics_list.values())
            final = vals[-1] if vals else {}
        else:
            final = {}
        names.append(dp_name.replace("_", " ").title())
        praucs.append(final.get("pr_auc", 0))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, praucs, color=['#66BB6A', '#FFA726', '#EF5350'], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('PR-AUC', fontsize=12)
    ax.set_title('Privacy–Utility Tradeoff (DP Noise Levels)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(praucs):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/figures/fig5_privacy_utility.png', dpi=150)
    plt.close(fig)
    print("    Saved fig5_privacy_utility.png")

# ============================================================
# Figure 6: Robustness
# ============================================================
def fig6_robustness():
    if not MPL_AVAILABLE:
        return
    print("  Fig 6: Robustness...")
    data = load_json("results/metrics/robustness.json")
    if data is None:
        return
    
    fedavg_pts, median_pts = [], []
    for exp_name, exp_data in data.items():
        agg = exp_data.get("aggregation", "")
        mal = exp_data.get("malicious_percent", 0)
        prauc = exp_data.get("final_pr_auc", 0)
        if agg == "FedAvg":
            fedavg_pts.append((mal, prauc))
        elif agg == "Median":
            median_pts.append((mal, prauc))
    
    fedavg_pts.sort()
    median_pts.sort()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if fedavg_pts:
        ax.plot([p[0] for p in fedavg_pts], [p[1] for p in fedavg_pts], 
                'o-', label='FedAvg', color='#2196F3', linewidth=2, markersize=8)
    if median_pts:
        ax.plot([p[0] for p in median_pts], [p[1] for p in median_pts], 
                's-', label='Median', color='#4CAF50', linewidth=2, markersize=8)
    ax.set_xlabel('Malicious Client %', fontsize=12)
    ax.set_ylabel('PR-AUC', fontsize=12)
    ax.set_title('Robustness Under Byzantine Attacks', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig('results/figures/fig6_robustness.png', dpi=150)
    plt.close(fig)
    print("    Saved fig6_robustness.png")

# ============================================================
# Figure 8: Static vs Drift-Aware
# ============================================================
def fig8_drift_adaptation():
    if not MPL_AVAILABLE:
        return
    print("  Fig 8: Static vs Drift-Aware...")
    path = "results/tables/Table_8_Drift_Adaptation.csv"
    if not os.path.exists(path):
        print("    [SKIP] No drift adaptation data.")
        return
    df = pd.read_csv(path)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['Window'], df['Static_PR_AUC'], 'o--', label='Static Model', color='#EF5350', linewidth=2)
    ax.plot(df['Window'], df['Adaptive_PR_AUC'], 's-', label='Drift-Aware Model', color='#4CAF50', linewidth=2)
    ax.set_xlabel('Temporal Window', fontsize=12)
    ax.set_ylabel('PR-AUC', fontsize=12)
    ax.set_title('Static vs Drift-Aware Adaptation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/figures/fig8_drift_adaptation.png', dpi=150)
    plt.close(fig)
    print("    Saved fig8_drift_adaptation.png")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 12: Generating Publication-Quality Figures")
    print("=" * 60)
    fig2_convergence()
    fig3_prauc_comparison()
    fig4_noniid()
    fig5_privacy()
    fig6_robustness()
    fig8_drift_adaptation()
    print("\n" + "=" * 60)
    print("Figure generation complete. See results/figures/")
    print("=" * 60)
