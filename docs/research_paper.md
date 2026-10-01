# TrustAML-FL: Privacy-Preserving Federated Heterogeneous Graph Learning for Cross-Bank Anti-Money Laundering Detection Under Non-IID Data and Temporal Drift

## 1. Abstract
Anti-Money Laundering (AML) detection in the financial sector requires bridging isolated data silos across multiple institutions without violating stringent data privacy regulations. This paper presents **TrustAML-FL**, a comprehensive framework that unites Federated Learning (FL) with Heterogeneous Graph Neural Networks (HGNNs). By constructing privacy-preserving federated heterogeneous graphs (modeling Accounts, Transactions, Devices, Locations) and evaluating under strict, methodologically sound constraints (Non-IID distributions, temporal out-of-time evaluation, and adversarial robustness), TrustAML-FL demonstrates that cross-bank collaboration can detect complex, multi-hop laundering typologies more effectively than isolated local models.

## 2. Introduction
Financial crime constitutes approximately 2-5% of global GDP. Current rule-based systems and isolated machine learning models suffer from severe false positive rates. Money laundering inherently spans multiple institutions, making it mathematically impossible for single-bank models to capture the full bipartite/tripartite laundering cycles (e.g., structuring, smurfing, and cross-border nested chains).

## 3. Motivation
Banks cannot share raw transaction data due to GDPR, CCPA, and bank secrecy laws. A federated framework allows models—not data—to traverse institutional boundaries. 

## 4. Related Work
Previous work on AML primarily focuses on centralized homogeneous graphs (e.g., Elliptic dataset) or simple tabular federated learning. Recent advances in FedGraph (Federated Graph Learning) often ignore realistic temporal drift or assume IID client distributions.

## 5. Research Gap
There is a lack of end-to-end evaluations of Federated HGNNs combining realistic non-IID conditions, strict chronological Out-of-Time (OOT) evaluation (preventing target leakage), and adversarial robustness in the financial domain.

## 6. Research Questions
- **RQ1**: How does federated heterogeneous graph learning compare with isolated bank-level models for AML detection?
- **RQ2**: How does institutional non-IID heterogeneity affect federated AML performance?
- **RQ3**: What privacy–utility tradeoff arises when differential privacy mechanisms are introduced?
- **RQ4**: How robust is federated AML detection against malicious client updates?
- **RQ5**: Can temporal drift detection and drift-aware adaptation reduce performance degradation?

## 7. Hypotheses
- **H1**: Federated learning improves AML detection compared with isolated local models under suitable cross-client conditions.
- **H2**: Increasing institutional non-IID heterogeneity reduces FedAvg performance.
- **H3**: FedProx can mitigate some degradation caused by client heterogeneity.
- **H4**: Increasing privacy noise produces a measurable privacy–utility tradeoff.
- **H5**: Robust aggregation (e.g., Coordinate-wise Median) can reduce degradation caused by malicious client updates.
- **H6**: Drift-aware adaptation reduces temporal performance degradation.

## 8. Proposed Framework
TrustAML-FL deploys PyTorch Geometric (PyG) HeteroData structures across decentralized clients. The Global server orchestrates FedAvg/FedProx aggregation while clients train localized GraphSAGE modules on their ego-networks.

## 9. Heterogeneous Graph Construction
Nodes: `Account`, `Transaction`, `Device`, `Location`.
Edges: `sends`, `receives`, `uses`, `located_at`.
Features are computed strictly dynamically from the observed temporal window to prevent future leakage (e.g., dynamic `transaction_count`).

## 10. Federated Learning
Clients execute localized message passing. We handle missing structural graph topologies across clients using lazy initialization mapping on the global server. 

## 11. Non-IID Setting
Institutions exhibit different transaction volume, feature distributions, and intrinsic AML rates. We synthesize this using Dirichlet distribution partitioning over features.

## 12. Differential Privacy
We implement gradient clipping and Gaussian noise injection. *Limitation*: Formal Epsilon ($\epsilon$) accounting is noted as `not_computed` due to technical limitations in applying RDP to decentralized graph message passing bounds.

## 13. Malicious Client Robustness
Threat model: A compromised bank sends malicious parameter updates (deltas) scaled to corrupt the global model. We evaluate Coordinate-wise Median aggregation as a defense mechanism against these delta-based attacks.

## 14. Temporal Drift
Financial topologies evolve. We enforce strict Out-of-Time (OOT) evaluation using non-overlapping temporal windows to accurately measure the model's performance on unseen future typologies.

## 15. Explainability
We approximate GNNExplainer using 1-hop ego-network traversals on actual predictions, generating JSON payloads containing critical feature contributions (amounts, locations) for human analysts.

## 16. Experimental Setup
Simulated 3-bank network using custom generated synthetic AML topologies (Structuring, Circular, Fan-out, Cross-border). Evaluated over seeds 42, 123, 456.

## 17. Results
(Refer to `results/tables/`)
- **Baseline**: Federated models generally surpass isolated local models in capturing cross-institution cycles.
- **Robustness**: Median aggregation successfully nullifies high-magnitude model replacement attacks.
- **Drift**: Adaptive continual learning (drift-aware retraining) recovers performance lost to temporal shifts.

## 18. Discussion
The framework successfully balances privacy and detection utility. Graph-based representation of banking features inherently requires more memory than tabular methods, highlighting a computational cost vs. topological intelligence tradeoff.

## 19. Threats to Validity
- **Construct Validity**: Synthetic data, while topological, lacks the true chaotic noise of real-world banking ecosystems.
- **Internal Validity**: Seed variance was tested, but highly skewed non-IID settings still cause instability in FL convergence.

## 20. Limitations
- Node features are dynamically aggregated but omit deep historical profiling for computational feasibility.
- DP Epsilon is uncalculated.
- Memory constraints scale with transaction volume, requiring subgraph sampling in production.

## 21. Conclusion
TrustAML-FL provides a rigorously tested, methodologically sound framework for federated AML detection, proving that multi-institutional topological analysis is possible without compromising raw data privacy.

## 22. Future Work
Integration of formal DP accounting for GNNs, secure multi-party computation (SMPC) for secure aggregation, and testing on real-world inter-bank transaction networks (e.g., SWIFT logs).

## 23. References
- Weber, M. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD.
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
