import sys
import os
import json
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.graph.builder import GraphBuilder
from src.models.hetero_graphsage import create_hetero_graphsage, PYG_AVAILABLE
from src.models.init import initialize_model
from src.explainability.alert_explanation import generate_alert_explanation
import torch.nn.functional as F

def run_explain():
    if not PYG_AVAILABLE:
        print("PyTorch Geometric not available. Exiting.")
        return
        
    print("Loading data for Explainability cases...")
    loader = DataLoader()
    tx_df, acc_df = loader.load_or_generate_data(num_banks=3, num_transactions=1500)
    
    preprocessor = DataPreprocessor()
    tx_df = preprocessor.fit_transform(tx_df, is_train=True)
    
    builder = GraphBuilder(tx_df, acc_df)
    data = builder.build()
    
    metadata = data.metadata()
    model = create_hetero_graphsage(metadata, hidden_channels=32, out_channels=1, num_layers=2)
    model = initialize_model(model, data)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        preds = torch.sigmoid(out['transaction']).numpy()
        y_true = data['transaction'].y.numpy()
        
    # Find TP, FP, FN, TN
    cases = {}
    
    for i in range(len(preds)):
        pred_prob = float(preds[i])
        pred_label = 1 if pred_prob > 0.5 else 0
        true_label = int(y_true[i])
        
        case_type = None
        if pred_label == 1 and true_label == 1 and "TP" not in cases:
            case_type = "TP"
        elif pred_label == 1 and true_label == 0 and "FP" not in cases:
            case_type = "FP"
        elif pred_label == 0 and true_label == 1 and "FN" not in cases:
            case_type = "FN"
        elif pred_label == 0 and true_label == 0 and "TN" not in cases:
            case_type = "TN"
            
        if case_type:
            # We must map back to original transaction ID to get features, but for simplicity
            # we just take the row since order is maintained in builder!
            row = tx_df.iloc[i]
            features = {
                "amount": float(row['amount']),
                "country": str(row['location']),
                "sender": str(row['sender_account'])
            }
            exp = generate_alert_explanation(
                transaction_id=str(row['transaction_id']),
                risk_score=float(pred_prob),
                features=features,
                aml_pattern=str(row.get('pattern', 'unknown'))
            )
            cases[case_type] = exp
            
        if len(cases) == 4:
            break
            
    os.makedirs("results", exist_ok=True)
    with open("results/explainability_cases.json", "w") as f:
        json.dump(cases, f, indent=4)
        
    print("Explainability cases generated and saved to results/explainability_cases.json")

if __name__ == "__main__":
    run_explain()
