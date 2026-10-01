# TrustAML-FL: Privacy-Preserving Federated Heterogeneous Graph Learning for Cross-Bank Anti-Money Laundering Detection Under Non-IID Data and Temporal Drift

*IEEE Conference Format Draft*

---

## Abstract
Anti-Money Laundering (AML) detection requires cross-institutional analysis of transaction networks, yet data privacy regulations prohibit direct data sharing between financial institutions. We present TrustAML-FL, a federated learning framework for AML detection that operates over heterogeneous transaction graphs. Our framework combines HeteroGraphSAGE models with FedAvg and FedProx aggregation, differential privacy mechanisms, and Byzantine-robust aggregation. We evaluate under controlled non-IID institutional heterogeneity, strict out-of-time temporal evaluation, and adversarial client attacks using synthetic AML transaction data with realistic laundering typologies. Results demonstrate that (1) federated learning achieves PR-AUC comparable to centralized training under IID conditions (0.84), (2) increasing institutional heterogeneity severely degrades performance (PR-AUC drops from 0.84 to 0.00 under severe non-IID), (3) coordinate-wise median aggregation maintains PR-AUC above 0.88 even under 30% malicious clients, and (4) drift-aware adaptation recovers up to 0.97 PR-AUC compared to 0.62 for static models. Epsilon accounting for differential privacy remains an open challenge for federated graph architectures.

**Keywords**: Anti-Money Laundering, Federated Learning, Heterogeneous Graphs, Graph Neural Networks, Privacy, Robustness, Non-IID, Temporal Drift

---

## I. Introduction
Financial crime detection is fundamentally a cross-institutional problem. Money laundering operations such as structuring, circular transfers, and cross-border layering inherently span multiple banks. However, financial institutions are legally prohibited from sharing raw transaction data under regulations including GDPR, CCPA, and various bank secrecy laws.

Federated Learning (FL) offers a paradigm where models — not data — traverse institutional boundaries. In this paper, we investigate whether FL applied to heterogeneous graph neural networks can improve AML detection compared to isolated local models, while preserving data privacy and maintaining robustness against adversarial participants.

## II. Related Work
Prior work on graph-based financial fraud detection primarily employs centralized homogeneous graphs [Weber et al., 2019] or simple tabular federated learning [Yang et al., 2019]. Recent federated graph learning literature [He et al., 2021] often assumes IID client distributions and does not evaluate temporal robustness.

## III. Problem Formulation
Let $\mathcal{B} = \{B_1, B_2, ..., B_K\}$ be $K$ financial institutions. Each institution holds a local heterogeneous graph $\mathcal{G}_k = (\mathcal{V}_k, \mathcal{E}_k)$ containing node types Account, Transaction, Device, and Location connected by typed edges (sends, receives, uses, located\_at). The objective is to train a shared model $\theta^*$ that classifies transaction nodes as AML-suspicious without sharing raw graph data.

## IV. Proposed Framework

### A. Heterogeneous Graph Construction
Node features are computed dynamically from the visible temporal window to prevent information leakage. Account features include transaction count, inbound/outbound volumes, and total amounts. Device and Location features capture usage frequency and transaction volumes.

### B. HeteroGraphSAGE Model
We employ GraphSAGE adapted for heterogeneous graphs via PyTorch Geometric's `to_hetero` transformation, with 2 message-passing layers and 32 hidden channels.

### C. Federated Aggregation
We evaluate FedAvg [McMahan et al., 2017] and FedProx [Li et al., 2020] with a proximal term $\mu = 0.01$.

### D. Differential Privacy
Client updates are protected via gradient clipping (norms: 0.1–5.0) and calibrated Gaussian noise (multipliers: 0.01–0.5). *Note: Formal $(\epsilon, \delta)$-DP accounting is not computed due to architectural constraints with heterogeneous graph message passing.*

### E. Byzantine Robustness
We defend against delta-based attacks using coordinate-wise median aggregation, which computes the element-wise median of client parameter updates rather than weighted averaging.

## V. Experimental Setup

### A. Dataset
Synthetic AML transaction data with 1,000 customers, 3 banks, and controlled injection of four AML typologies: structuring, circular transfers, fan-out, and cross-border chains. AML rate: ~30%.

### B. Evaluation
- Strict out-of-time (OOT) evaluation with expanding temporal windows
- Multi-seed validation (seeds: 42, 123, 456)
- Primary metric: PR-AUC (Average Precision)

## VI. Results

### A. RQ1: Federated vs Local
Under IID conditions, FedAvg achieves mean PR-AUC = 0.835 (std = 0.038), comparable to centralized training.

### B. RQ2: Non-IID Impact
| Degree | FedAvg PR-AUC | FedProx PR-AUC |
|--------|---------------|----------------|
| IID | 0.835 ± 0.038 | 0.826 ± 0.037 |
| Moderate | 0.564 ± 0.036 | 0.481 ± 0.230 |
| Severe | 0.000 ± 0.000 | 0.000 ± 0.000 |

### C. RQ3: Privacy–Utility Tradeoff
Increasing noise multiplier from 0.01 to 0.5 produces measurable PR-AUC degradation. Formal epsilon remains uncomputed.

### D. RQ4: Robustness
| Attack % | FedAvg PR-AUC | Median PR-AUC |
|----------|---------------|---------------|
| 0% | 0.984 | 0.879 |
| 10% | 0.914 | 0.949 |
| 20% | 0.866 | 0.915 |
| 30% | 0.210 | 0.883 |

### E. RQ5: Drift Adaptation
Adaptive retraining achieves PR-AUC 0.87–0.97 across temporal windows versus 0.62–0.88 for static models.

## VII. Threats to Validity
- Synthetic data lacks real-world noise characteristics
- 3-client federation is smaller than production consortia
- Severe non-IID collapse may be an artifact of small partition sizes

## VIII. Conclusion
TrustAML-FL demonstrates that federated heterogeneous graph learning is viable for cross-bank AML detection under controlled conditions. Robustness to adversarial clients and drift-aware adaptation are confirmed strengths. Severe non-IID heterogeneity and formal privacy accounting remain open challenges.

## References
[1] B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017.
[2] T. Li et al., "Federated Optimization in Heterogeneous Networks," MLSys, 2020.
[3] M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks," KDD, 2019.
