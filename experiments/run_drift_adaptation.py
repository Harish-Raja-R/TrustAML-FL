import sys
import os
import json
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.evaluation.temporal import TemporalEvaluator
from src.graph.builder import GraphBuilder
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.models.init import initialize_model
from src.evaluation.metrics import evaluate_predictions
import torch.nn.functional as F

def train_local(model, data, epochs=5, lr=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y = data['transaction'].y
    pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else torch.tensor(1.0)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy(out['transaction'], y.float(), weight=torch.where(y == 1, pos_weight, 1.0))
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

def run_adaptation():
    if not PYG_AVAILABLE:
        return
        
    print("Loading data for Drift Adaptation...")
    loader = DataLoader()
    tx_df, acc_df = loader.load_or_generate_data(num_banks=3, num_transactions=1500)
    
    from src.data.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    tx_df = preprocessor.fit_transform(tx_df, is_train=True)
    
    evaluator = TemporalEvaluator(tx_df, num_windows=5)
    folds = evaluator.get_oot_folds()
    
    # We will simulate a Global Static Model vs a Drift-Aware (Retrained) Model
    # Since FL takes time, we simulate the "drift adaptation" locally for speed but concept applies.
    
    # Train Static Model on Window 1
    train_builder = GraphBuilder(folds[0]['train'], acc_df)
    train_data = train_builder.build()
    
    metadata = train_data.metadata()
    static_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
    static_model = initialize_model(static_model, train_data)
    static_model = train_local(static_model, train_data, epochs=15)
    
    adaptive_model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
    adaptive_model.load_state_dict(static_model.state_dict())
    
    results = []
    
    for fold in folds:
        test_builder = GraphBuilder(fold['test'], acc_df)
        test_data = test_builder.build()
        
        static_metrics = evaluate_model(static_model, test_data)
        
        # Adaptive model evaluates first on new window...
        adaptive_metrics_before = evaluate_model(adaptive_model, test_data)
        
        # ...then retrains (adapts) on this new window for the next iteration (simulated continuous learning)
        # We actually train it on the current fold's train data (which includes new info)
        current_train_builder = GraphBuilder(fold['train'], acc_df)
        current_train_data = current_train_builder.build()
        adaptive_model = train_local(adaptive_model, current_train_data, epochs=5)
        
        # Now evaluate it after adaptation (just to see if it recovered)
        adaptive_metrics_after = evaluate_model(adaptive_model, test_data)
        
        results.append({
            "Window": fold['fold_id'],
            "Static_PR_AUC": static_metrics['pr_auc'],
            "Adaptive_PR_AUC": adaptive_metrics_after['pr_auc']
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/Table_8_Drift_Adaptation.csv", index=False)
    print("Drift Adaptation completed.")
    print(df)

if __name__ == "__main__":
    run_adaptation()
