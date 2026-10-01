import pandas as pd
import numpy as np

class DataPartitioner:
    def __init__(self, tx_df, num_clients=3, non_iid_degree="iid"):
        """
        non_iid_degree: 'iid', 'moderate', 'severe'
        """
        self.tx_df = tx_df
        self.num_clients = num_clients
        self.non_iid_degree = non_iid_degree
        
    def partition(self):
        print(f"Partitioning data into {self.num_clients} clients (Degree: {self.non_iid_degree})")
        
        # Base partitioning by actual sender_bank if available and corresponds to num_clients
        banks = self.tx_df['sender_bank'].unique()
        if len(banks) == self.num_clients and self.non_iid_degree == "iid":
            # If already assigned randomly, might be IID enough, but let's strictly partition by bank
            partitions = {bank: self.tx_df[self.tx_df['sender_bank'] == bank] for bank in banks}
            return partitions
            
        partitions = {}
        
        if self.non_iid_degree == "iid":
            # Pure random split
            shuffled = self.tx_df.sample(frac=1).reset_index(drop=True)
            splits = np.array_split(shuffled, self.num_clients)
            for i in range(self.num_clients):
                partitions[f"Client_{i}"] = splits[i]
                
        elif self.non_iid_degree == "moderate":
            # Partition by transaction type preference
            types = self.tx_df['transaction_type'].unique()
            # Try to assign a dominant type to each client
            splits = {f"Client_{i}": pd.DataFrame() for i in range(self.num_clients)}
            
            for i, tx_type in enumerate(types):
                type_data = self.tx_df[self.tx_df['transaction_type'] == tx_type]
                primary_client = i % self.num_clients
                
                # 70% to primary client, 30% split among others
                mask = np.random.rand(len(type_data)) < 0.7
                splits[f"Client_{primary_client}"] = pd.concat([splits[f"Client_{primary_client}"], type_data[mask]])
                
                remaining = type_data[~mask]
                if len(remaining) > 0:
                    other_clients = [c for c in range(self.num_clients) if c != primary_client]
                    rem_splits = np.array_split(remaining, len(other_clients))
                    for j, c in enumerate(other_clients):
                        splits[f"Client_{c}"] = pd.concat([splits[f"Client_{c}"], rem_splits[j]])
                        
            partitions = splits
            
        elif self.non_iid_degree == "severe":
            # Extreme partitioning based on transaction type (e.g., each bank only sees certain types)
            types = self.tx_df['transaction_type'].unique()
            splits = {f"Client_{i}": pd.DataFrame() for i in range(self.num_clients)}
            
            for i, tx_type in enumerate(types):
                type_data = self.tx_df[self.tx_df['transaction_type'] == tx_type]
                primary_client = i % self.num_clients
                
                # 95% to primary client
                mask = np.random.rand(len(type_data)) < 0.95
                splits[f"Client_{primary_client}"] = pd.concat([splits[f"Client_{primary_client}"], type_data[mask]])
                
                remaining = type_data[~mask]
                if len(remaining) > 0:
                    other_clients = [c for c in range(self.num_clients) if c != primary_client]
                    rem_splits = np.array_split(remaining, len(other_clients))
                    for j, c in enumerate(other_clients):
                        splits[f"Client_{c}"] = pd.concat([splits[f"Client_{c}"], rem_splits[j]])
            
            # Also skew the class imbalance (fraud rates)
            for i in range(self.num_clients):
                fraud = splits[f"Client_{i}"][splits[f"Client_{i}"]['is_aml'] == 1]
                legit = splits[f"Client_{i}"][splits[f"Client_{i}"]['is_aml'] == 0]
                
                # Modulate fraud
                keep_fraud = np.random.uniform(0.1, 1.0)
                mask_f = np.random.rand(len(fraud)) < keep_fraud
                splits[f"Client_{i}"] = pd.concat([fraud[mask_f], legit])
                
            partitions = splits
            
        return partitions

if __name__ == "__main__":
    tx_df = pd.read_csv("data/synthetic/transactions.csv")
    partitioner = DataPartitioner(tx_df, non_iid_degree="severe")
    parts = partitioner.partition()
    for k, v in parts.items():
        print(f"{k}: {len(v)} transactions, AML: {v['is_aml'].sum()}")
