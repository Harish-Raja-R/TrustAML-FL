import pandas as pd
import numpy as np

class TemporalEvaluator:
    def __init__(self, tx_df, num_windows=5):
        # Sort transactions chronologically
        self.tx_df = tx_df.sort_values(by="timestamp").reset_index(drop=True)
        self.num_windows = num_windows
        self.windows = np.array_split(self.tx_df, self.num_windows)
        
    def get_oot_folds(self):
        """
        Returns temporal folds. 
        Fold 1: Train = Window 1, Test = Window 2
        Fold 2: Train = Window 1+2, Test = Window 3
        Fold 3: Train = Window 1+2+3, Test = Window 4
        """
        folds = []
        train_dfs = []
        for i in range(self.num_windows - 1):
            train_dfs.append(self.windows[i])
            train_df_concat = pd.concat(train_dfs).reset_index(drop=True)
            test_df = self.windows[i+1].reset_index(drop=True)
            folds.append({
                'fold_id': i + 1,
                'train': train_df_concat,
                'test': test_df
            })
        return folds
