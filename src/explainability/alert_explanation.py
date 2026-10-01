def generate_alert_explanation(transaction_id, risk_score, features, aml_pattern=None):
    """
    Generates human-readable explanation for an AML alert based on features.
    """
    reasons = []
    
    amount = features.get("amount", 0)
    channel = features.get("channel", "unknown")
    country = features.get("country", "US")
    
    if amount > 50000:
        reasons.append("Unusually high transaction amount.")
        
    if country in ["KY", "UAE", "SG"]:
        reasons.append("Cross-border transaction to high-risk jurisdiction.")
        
    if aml_pattern and aml_pattern != "normal":
        reasons.append(f"Suspicious topological pattern detected: {aml_pattern}.")
        
    if not reasons:
        reasons.append("Elevated risk score from ML model.")
        
    explanation = {
        "transaction_id": transaction_id,
        "risk_score": float(risk_score),
        "risk_level": "CRITICAL" if risk_score > 0.9 else "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.5 else "LOW",
        "reasons": reasons,
        "pattern": aml_pattern
    }
    
    return explanation
