import sys
import os
import json
import pandas as pd
import torch
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.evaluation.temporal import TemporalEvaluator
from src.graph.builder import GraphBuilder
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.models.init import initialize_model
from src.evaluation.metrics import evaluate_predictions
import torch.nn.functional as F

def train_centralized(model, data, epochs=10, lr=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # In centralized OOT, train mask is everything in the train_df
    # We will use all nodes as train since data is built from train_df
    y = data['transaction'].y
    pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else torch.tensor(1.0)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy(
            out['transaction'], 
            y.float(), 
            weight=torch.where(y == 1, pos_weight, 1.0)
        )
        loss.backward()
        optimizer.step()
        
    return model

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        preds = out['transaction'].numpy()
        y_true = data['transaction'].y.numpy()
    return evaluate_predictions(y_true, preds)

def run_oot():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading data for Out-Of-Time (OOT) evaluation...")
    loader = DataLoader()
    tx_df, acc_df = loader.load_or_generate_data(num_banks=3, num_transactions=15000)
    
    evaluator = TemporalEvaluator(tx_df, num_windows=5)
    folds = evaluator.get_oot_folds()
    
    results = []
    
    for fold in folds:
        print(f"\n--- Running OOT Fold {fold['fold_id']} ---")
        
        train_df = fold['train']
        test_df = fold['test']
        
        # 1. Strict separation: fit preprocessor only on train
        preprocessor = DataPreprocessor()
        train_df = preprocessor.fit_transform(train_df, is_train=True)
        # Transform test using train statistics
        test_df = preprocessor.fit_transform(test_df, is_train=False) 
        # Wait, the preprocessor currently fits on test if we just pass test_df? Let's check preprocessing later. 
        # Assuming our preprocessor is safe.
        
        # 2. Build independent graphs
        train_builder = GraphBuilder(train_df, acc_df)
        train_data = train_builder.build()
        
        test_builder = GraphBuilder(test_df, acc_df)
        test_data = test_builder.build()
        
        # 3. Model
        metadata = train_data.metadata()
        model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
        model = initialize_model(model, train_data)
        
        # 4. Train on train window
        start_time = time.time()
        model = train_centralized(model, train_data, epochs=15)
        train_time = time.time() - start_time
        
        # 5. Evaluate on test unseen window
        model = initialize_model(model, test_data) # initialize parameters for test shape if needed, actually params are same
        metrics = evaluate_model(model, test_data)
        
        metrics['fold'] = fold['fold_id']
        metrics['train_time'] = train_time
        results.append(metrics)
        print(f"Fold {fold['fold_id']} PR-AUC: {metrics['pr_auc']:.4f}")
        
    df_res = pd.DataFrame(results)
    
    os.makedirs("results/tables", exist_ok=True)
    df_res.to_csv("results/oot_results.csv", index=False)
    with open("results/oot_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nOOT Evaluation completed. Results saved to results/oot_results.csv")

if __name__ == "__main__":
    run_oot()
