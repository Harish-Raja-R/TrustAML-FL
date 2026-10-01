from fastapi import APIRouter
import pandas as pd
import os
import random
from src.explainability.alert_explanation import generate_alert_explanation

router = APIRouter()

@router.get("/")
def get_alerts(limit: int = 10):
    tx_file = "data/synthetic/transactions.csv"
    if not os.path.exists(tx_file):
        return []
        
    df = pd.read_csv(tx_file)
    # Get some AML transactions
    aml_tx = df[df['is_aml'] == 1]
    
    if aml_tx.empty:
        # Fallback if no AML generated
        aml_tx = df.head(limit)
    else:
        aml_tx = aml_tx.head(limit)
        
    alerts = []
    for _, row in aml_tx.iterrows():
        # Fake risk score since we don't have predictions saved
        # In a real system, this comes from model inference DB
        risk_score = random.uniform(0.7, 0.99) if row['is_aml'] == 1 else random.uniform(0.1, 0.4)
        
        features = {
            "amount": row['amount'],
            "country": "US" # simplified
        }
        
        alert = generate_alert_explanation(
            transaction_id=row['tx_id'],
            risk_score=risk_score,
            features=features,
            aml_pattern=row['pattern_type'] if 'pattern_type' in row else "unknown"
        )
        
        alert['bank'] = row['bank_id']
        alert['timestamp'] = row['timestamp']
        
        alerts.append(alert)
        
    return alerts
