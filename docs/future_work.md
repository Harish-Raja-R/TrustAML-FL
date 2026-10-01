# Future Work

## 1. Real-World Data Validation
Evaluate TrustAML-FL on anonymized real banking transaction datasets (e.g., from SWIFT, FinCEN, or partner institutions) to validate generalizability beyond synthetic topologies.

## 2. Formal Differential Privacy with RDP Accountant
Integrate Opacus or a custom Rényi Differential Privacy (RDP) accountant compatible with PyTorch Geometric's heterogeneous message passing to provide rigorous (ε, δ)-DP guarantees.

## 3. Secure Multi-Party Computation (SMPC)
Replace plaintext parameter aggregation with SMPC protocols (e.g., secret sharing, homomorphic encryption) to prevent model inversion and parameter inference attacks.

## 4. Scalability Testing
Evaluate with 50–100+ federated clients to assess communication overhead, convergence behavior, and aggregation scalability in realistic consortia.

## 5. FedProx Hyperparameter Optimization
Conduct systematic grid or Bayesian search over the proximal term μ to determine optimal settings for heterogeneous banking data. Explore alternative FL algorithms (FedNova, SCAFFOLD, FedBN).

## 6. Advanced GNN Architectures
Replace or augment GraphSAGE with attention-based architectures (GAT, HGT — Heterogeneous Graph Transformer) for potentially improved cross-type message passing.

## 7. Full GNNExplainer Integration
Implement GNNExplainer or PGExplainer for rigorous subgraph-level attribution, replacing the current 1-hop approximation.

## 8. Online Drift Detection and Automated Adaptation
Implement online concept drift detectors (e.g., ADWIN, DDM) that trigger model retraining automatically when distribution shift exceeds a threshold.

## 9. Regulatory Compliance Framework
Map the system's outputs to specific regulatory requirements (FATF Recommendations, EU 6AMLD, BSA/AML) and design compliance-aware alert generation.

## 10. Production Engineering
Containerize the system (Docker/Kubernetes), add monitoring (Prometheus/Grafana), implement a production-grade API gateway, and design audit trails for regulatory inspection.
