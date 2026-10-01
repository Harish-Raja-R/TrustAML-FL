# Reproducibility Guide

To reproduce the findings in the TrustAML-FL Phase 12 evaluation, follow these instructions precisely.

## 1. Environment Setup
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## 2. Seed Configuration
All experiments are designed to run deterministically. 
In Python, we initialize the seeds centrally:
```python
from src.utils.seed import set_seed
set_seed(42)  # Seeds for torch, numpy, and random
```

## 3. Running the Comprehensive Pipeline
To execute the baseline models (Centralized, Local, FedAvg, FedProx, Privacy, Robustness, Temporal Drift):
```bash
python experiments/run_full_pipeline.py
```
This automatically dumps all raw JSON metrics into `results/metrics/`.

## 4. Phase 12 Specific Validations
To reproduce the strict Phase 12 claims:
1. **Out of Time Evaluation**:
   ```bash
   python experiments/run_oot.py
   ```
2. **Drift Adaptation (Static vs Adaptive)**:
   ```bash
   python experiments/run_drift_adaptation.py
   ```
3. **Non-IID Multi-Seed Matrices (42, 123, 456)**:
   ```bash
   python experiments/run_noniid_seeds.py
   ```
4. **Explainability Extraction**:
   ```bash
   python experiments/run_explainability.py
   ```

## 5. Generating Publication Tables
Once the evaluations finish writing JSONs, compile them into CSVs and Markdown tables using:
```bash
python experiments/generate_final_tables.py
```
Output tables will be saved into `results/tables/`.

## 6. Testing
```bash
python -m pytest -q
```
We have explicitly included tests asserting that:
1. The Malicious Client parameter scaling modifies the actual gradient delta.
2. The PyTorch Geometric `HeteroData` lazy module structure successfully initializes across distributed networks.
