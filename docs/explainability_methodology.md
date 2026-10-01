# Explainability Methodology

In the TrustAML-FL framework, model explainability is crucial for regulatory compliance and analyst efficiency. We implement a post-hoc explainability module that interprets the predictions made by our Federated Heterogeneous Graph Neural Network (HeteroGraphSAGE).

## Process
1. **Prediction Extraction**: During evaluation, the model outputs a probability (risk score) for each transaction node.
2. **Thresholding**: We define a static threshold (e.g., 0.5) to classify transactions as Suspicious (Alert) or Normal.
3. **Feature Attribution**: The explainer extracts the original transaction features (amount, currency, channel) and topological properties.
4. **Subgraph Context**: (Approximation) Instead of deploying full GNNExplainer due to computational complexity in cross-bank settings, we approximate the explanation by analyzing the 1-hop ego-network of the transaction (the sender account, receiver account, and device used).
5. **Human-Readable Alert**: The system translates these mathematical properties into a human-readable JSON payload containing `risk_level`, `reasons`, and `features`.

## Generated Cases
See `results/explainability_cases.json` for real predictions mapping to True Positives, False Positives, False Negatives, and True Negatives.
