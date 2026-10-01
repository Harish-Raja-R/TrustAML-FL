from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class BaselineClassifier:
    def __init__(self, model_type="lr", seed=42):
        self.model_type = model_type
        if model_type == "lr":
            self.model = LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced')
        elif model_type == "rf":
            self.model = RandomForestClassifier(random_state=seed, n_estimators=100, class_weight='balanced')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def predict(self, X):
        return self.model.predict(X)
