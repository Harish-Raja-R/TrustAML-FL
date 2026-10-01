# Limitations

This document records the known limitations of the TrustAML-FL framework as of Phase 12.

## 1. Synthetic Data Only
All experiments use synthetically generated transaction data. While the generator includes realistic AML typologies (structuring, circular transfers, fan-out, cross-border chains), the statistical properties differ from real banking ecosystems. Real-world validation is required before any deployment conclusions.

## 2. Memory Constraints
PyTorch Geometric heterogeneous graph operations on CPU require substantial memory. Experiments were scaled down from 20,000 to 1,500 transactions per bank to fit within available RAM. This limits the statistical power of some results (e.g., severe Non-IID with very small client partitions).

## 3. Formal Differential Privacy Accounting
Epsilon (ε) is reported as `not_computed`. The system implements gradient clipping and Gaussian noise injection, but a validated Rényi Differential Privacy (RDP) accountant for federated heterogeneous graph message passing is not integrated. This means no formal (ε, δ)-DP guarantee can be claimed.

## 4. Severe Non-IID Collapse
Under severe Non-IID partitioning, both FedAvg and FedProx produce PR-AUC = 0.00. This is a genuine result: when the partitioning is extreme enough, some clients may have zero positive labels in their local data, making meaningful training impossible. Production deployments would need minimum label guarantees or pre-filtering.

## 5. FedProx Performance
FedProx did not consistently outperform FedAvg in our experiments, and showed higher variance under moderate Non-IID conditions. This may be due to the limited number of clients (3), small number of rounds (5), or the proximal term (μ = 0.01) being suboptimal. Hyperparameter tuning of μ was not exhaustively explored.

## 6. Limited Explainability
We approximate graph-level explanation using 1-hop ego-network analysis rather than full GNNExplainer, due to computational cost. The resulting explanations are informative but do not provide guaranteed optimal subgraph attributions.

## 7. Number of Clients
Experiments use 3 federated clients (banks), with up to 10 for robustness testing. Real-world consortia may involve hundreds of institutions with vastly different scales.

## 8. Temporal Evaluation Windows
OOT evaluation uses 5 chronological windows. With 1,500 base transactions per bank, each window is relatively small, which increases metric variance across folds.

## 9. No Secure Aggregation
The federation protocol transmits model parameters in plaintext (simulated). Production deployment would require secure aggregation (e.g., SMPC, homomorphic encryption) to prevent parameter inference attacks.

## 10. No Real Regulatory Compliance Testing
The framework has not been validated against any specific regulatory framework (e.g., FATF, EU AMLD, BSA/AML). Compliance claims are not made.
