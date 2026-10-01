# TrustAML-FL: Privacy-Preserving Federated Heterogeneous Graph Learning for Cross-Bank AML Detection

## Abstract
This paper introduces TrustAML-FL, a novel methodology for detecting anti-money laundering (AML) patterns across distributed financial institutions using Federated Heterogeneous Graph Learning.

## 1. Methodology
We implement a heterogeneous graph schema encompassing transaction, account, device, and location entities. Models are trained using federated aggregations (FedAvg and FedProx) across banks with varying data distributions (IID and Non-IID).

## 2. Experiments and Results
Our experiments show the following baseline achievements:
- **Baseline Utility:** Measured using Centralized vs FedAvg (see `results/metrics/table_robustness.csv` when generated).
- **Differential Privacy:** Not formally accounted (privacy_accounting_status = "not_computed"). Epsilon guarantees are not claimed. We evaluated the empirical utility tradeoff under various clipping norms and noise multipliers.
- **Robustness:** Our experiments show that under Malicious Client attacks operating on update deltas, standard FedAvg collapses, while coordinate-wise Median aggregation successfully defends against synthetic adversaries.
- **Temporal Drift:** We distinguish between "synthetic injected drift" and "natural temporal distribution shift" (see `table_drift.csv`).

## 3. Security Guarantee Limitations
The system currently employs a "secure aggregation simulation" for conceptual evaluation. It is not a production cryptographic secure aggregation mechanism.

## 4. Conclusion
TrustAML-FL provides a comprehensive evaluation environment for privacy-preserving AML detection.
