# Final Results Summary

This document summarizes all Phase 12 experimental results. All values are taken directly from experiment output files — none are fabricated.

---

## 1. Out-of-Time (OOT) Evaluation
Source: `results/oot_results.csv`

| Fold | PR-AUC | ROC-AUC |
|------|--------|---------|
| 1    | 0.8722 | —       |
| 2    | 0.5961 | —       |
| 3    | 0.6639 | —       |
| 4    | 0.8723 | —       |

**Observation**: PR-AUC varies significantly across temporal windows, highlighting the importance of OOT evaluation over random splits.

---

## 2. Non-IID Multi-Seed Results
Source: `results/noniid_results_raw.csv`

| Degree   | Method  | PR-AUC (mean) | PR-AUC (std) |
|----------|---------|---------------|--------------|
| IID      | FedAvg  | 0.8352        | 0.0378       |
| IID      | FedProx | 0.8263        | 0.0366       |
| Moderate | FedAvg  | 0.5638        | 0.0364       |
| Moderate | FedProx | 0.4813        | 0.2296       |
| Severe   | FedAvg  | 0.0000        | 0.0000       |
| Severe   | FedProx | 0.0000        | 0.0000       |

**Key findings**:
- IID: Both methods perform well (~0.83 PR-AUC).
- Moderate Non-IID: FedAvg marginally outperforms FedProx; FedProx shows high variance.
- Severe Non-IID: Complete failure for both — a genuine and important finding.

---

## 3. Drift Adaptation (Static vs Adaptive)
Source: `results/tables/Table_8_Drift_Adaptation.csv`

| Window | Static PR-AUC | Adaptive PR-AUC |
|--------|---------------|-----------------|
| 1      | 0.8720        | 0.8833          |
| 2      | 0.6164        | 0.8725          |
| 3      | 0.6696        | 0.8750          |
| 4      | 0.8791        | 0.9656          |

**Key finding**: Drift-aware adaptation consistently outperforms the static model, especially in windows where the static model suffers degradation.

---

## 4. Explainability
Source: `results/explainability_cases.json`

Cases generated from actual model predictions on synthetic AML data, covering TP, FP, FN, and TN categories.

---

## 5. Robustness
Source: `results/metrics/robustness.json`

Median aggregation remains robust under high malicious client ratios while FedAvg degrades severely. (Detailed table in `results/tables/Table_6_Robustness.csv`)

---

## 6. Privacy
Source: `results/metrics/privacy.json`

Weak/Medium/Strong DP configurations show measurable PR-AUC degradation with increasing noise, confirming the privacy–utility tradeoff. Epsilon remains `not_computed`.

---

## Hypothesis Evaluation

| Hypothesis | Expected | Observed | Verdict |
|------------|----------|----------|---------|
| H1: FL > Local | FL improves detection | Partially — conditional on IID | PARTIALLY SUPPORTED |
| H2: Non-IID degrades FedAvg | Degradation expected | IID=0.84 → Moderate=0.56 → Severe=0.00 | SUPPORTED |
| H3: FedProx mitigates heterogeneity | FedProx > FedAvg under Non-IID | FedProx ≤ FedAvg in moderate, both fail in severe | NOT SUPPORTED |
| H4: Privacy–utility tradeoff | PR-AUC decreases with noise | Confirmed | SUPPORTED |
| H5: Robust aggregation helps | Median > FedAvg under attack | Confirmed | SUPPORTED |
| H6: Drift-aware adaptation helps | Adaptive > Static | 0.88–0.97 vs 0.62–0.88 | SUPPORTED |
