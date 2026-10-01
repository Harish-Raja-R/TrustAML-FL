# Experimental Methodology

## 1. Overview
This document describes the experimental design and methodology used in TrustAML-FL Phase 12 evaluations. All experiments use synthetic AML transaction data generated deterministically from controlled seeds.

## 2. Data Generation
- **Generator**: `src/data/generator.py` (`AMLDataGenerator`)
- **Entity types**: Customers, Accounts, Merchants, Devices, Locations
- **AML Patterns**: Structuring (smurfing), Circular transfers, Fan-out (mule behavior), Cross-border chains
- **Base parameters**: 1,000 customers, 1,500 base transactions, 3 banks
- **Seed**: 42 (default), also tested with 123 and 456

## 3. Preprocessing
- **Categorical encoding**: `LabelEncoder` for currency, transaction_type, channel
- **Numerical scaling**: `StandardScaler` for amount
- **Strict separation**: Preprocessor is `fit()` only on training data; test data uses `transform()` only
- **Implementation**: `src/data/preprocessing.py`

## 4. Graph Construction
- **Framework**: PyTorch Geometric `HeteroData`
- **Node types**: Account, Transaction, Device, Location
- **Edge types**: sends, receives, uses, located_at
- **Node features**: Dynamically computed from visible temporal window only (no future leakage)
  - Account: [transaction_count, inbound_count, outbound_count, total_inbound_amount, total_outbound_amount]
  - Device: [transaction_count, total_amount]
  - Location: [transaction_count, total_amount]
  - Transaction: [amount, currency_encoded, type_encoded, channel_encoded]

## 5. Model Architecture
- **Type**: HeteroGraphSAGE (GraphSAGE adapted for heterogeneous graphs via PyG `to_hetero`)
- **Hidden channels**: 32
- **Output**: 1 (binary AML probability per transaction node)
- **Layers**: 2
- **Activation**: Sigmoid
- **Loss**: Binary Cross-Entropy with class-weighted positive samples

## 6. Federated Learning Configuration
- **Algorithm**: FedAvg, FedProx (μ = 0.01)
- **Clients**: 3 banks (default), 10 for robustness
- **Rounds**: 5–15 depending on experiment
- **Local epochs**: 1–2
- **Learning rate**: 0.01
- **Aggregation**: FedAvg (weighted by sample count), Coordinate-wise Median

## 7. Out-of-Time (OOT) Evaluation
- **Implementation**: `src/evaluation/temporal.py`
- **Method**: Expanding-window temporal folds
- **Windows**: 5 chronological splits
- **Constraint**: Test window never influences preprocessing, graph statistics, or training

## 8. Non-IID Heterogeneity
- **Degrees**: IID, Moderate, Severe
- **Implementation**: `src/data/partition.py` (Dirichlet-based partitioning)
- **Seeds**: 42, 123, 456 (3 independent runs per condition)
- **Reporting**: Mean ± standard deviation

## 9. Differential Privacy
- **Mechanism**: Gradient clipping + Gaussian noise injection
- **Configurations**: Weak (clip=5.0, noise=0.01), Medium (clip=1.0, noise=0.1), Strong (clip=0.1, noise=0.5)
- **Epsilon accounting**: `not_computed` (documented limitation)

## 10. Robustness Testing
- **Attack model**: Delta-based (modifies `local_model - global_model` updates, not absolute weights)
- **Attack types**: Model replacement (scaling attack, scale=20.0)
- **Malicious ratios**: 0%, 10%, 20%, 30%
- **Defenses**: FedAvg, Coordinate-wise Median

## 11. Temporal Drift
- **Synthetic injected**: Explicit perturbation of a controlled temporal window
- **Natural windows**: Untouched chronological data split into windows
- **Metrics**: Wasserstein distance, KL divergence, AML rate shift

## 12. Drift-Aware Adaptation
- **Static model**: Trained on Window 1 only, evaluated on all subsequent windows
- **Adaptive model**: Retrained (continual learning) on each new window before evaluation on the next

## 13. Explainability
- **Method**: Post-hoc 1-hop ego-network approximation (not full GNNExplainer)
- **Input**: Actual model predictions (not fabricated)
- **Output**: JSON cases for TP, FP, FN, TN with risk scores and feature attributions

## 14. Primary Metric
- **PR-AUC** (Average Precision) — standard for highly imbalanced AML detection
- Secondary: ROC-AUC, F1, MCC, Precision, Recall, FPR, FNR
