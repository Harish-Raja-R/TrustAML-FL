import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def fit_transform(self, df, is_train=True):
        processed = df.copy()
        
        # Categorical features
        cat_cols = ['currency', 'transaction_type', 'channel']
        for col in cat_cols:
            if col in processed.columns:
                if is_train:
                    self.encoders[col] = LabelEncoder()
                    processed[col] = self.encoders[col].fit_transform(processed[col].astype(str))
                else:
                    # Handle unseen labels gracefully (assign to a default or -1)
                    classes = self.encoders[col].classes_
                    processed[col] = processed[col].apply(lambda x: x if x in classes else 'UNKNOWN')
                    # Update classes to include UNKNOWN if not present
                    if 'UNKNOWN' not in classes:
                        classes = np.append(classes, 'UNKNOWN')
                        self.encoders[col].classes_ = classes
                    processed[col] = self.encoders[col].transform(processed[col].astype(str))
                    
        # Numerical features
        num_cols = ['amount']
        for col in num_cols:
            if col in processed.columns:
                if is_train:
                    self.scalers[col] = StandardScaler()
                    processed[col] = self.scalers[col].fit_transform(processed[[col]])
                else:
                    processed[col] = self.scalers[col].transform(processed[[col]])
                    
        return processed
        
    def transform(self, df):
        return self.fit_transform(df, is_train=False)
