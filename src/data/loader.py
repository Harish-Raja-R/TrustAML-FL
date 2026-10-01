import pandas as pd
import torch
import os
from src.data.generator import AMLDataGenerator
from src.data.preprocessing import DataPreprocessor
from src.data.partition import DataPartitioner
from src.graph.builder import GraphBuilder

class DataLoader:
    def __init__(self, data_dir="data/synthetic"):
        self.data_dir = data_dir
        
    def load_or_generate_data(self, force_generate=False, num_banks=3, num_transactions=20000):
        tx_path = os.path.join(self.data_dir, "transactions.csv")
        acc_path = os.path.join(self.data_dir, "accounts.csv")
        
        if force_generate or not (os.path.exists(tx_path) and os.path.exists(acc_path)):
            os.makedirs(self.data_dir, exist_ok=True)
            generator = AMLDataGenerator(num_banks=num_banks, num_transactions=num_transactions)
            tx_df, acc_df = generator.generate()
            tx_df.to_csv(tx_path, index=False)
            acc_df.to_csv(acc_path, index=False)
        else:
            tx_df = pd.read_csv(tx_path)
            acc_df = pd.read_csv(acc_path)
            
        return tx_df, acc_df

    def get_centralized_graph(self):
        tx_df, acc_df = self.load_or_generate_data()
        
        preprocessor = DataPreprocessor()
        tx_df = preprocessor.fit_transform(tx_df, is_train=True)
        
        builder = GraphBuilder(tx_df, acc_df)
        hetero_data = builder.build()
        
        return hetero_data, preprocessor

    def get_federated_graphs(self, num_clients=3, non_iid_degree="iid"):
        tx_df, acc_df = self.load_or_generate_data(num_banks=num_clients)
        
        # Partition data first
        partitioner = DataPartitioner(tx_df, num_clients=num_clients, non_iid_degree=non_iid_degree)
        partitions = partitioner.partition()
        
        client_graphs = {}
        # We need to share the same preprocessor (fit on global) for realistic feature alignment
        # Or fit independently to simulate realistic isolation. 
        # In this prototype, fitting globally then transforming is standard for research baselines.
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(tx_df, is_train=True)
        
        for client_id, client_tx_df in partitions.items():
            processed_tx = preprocessor.transform(client_tx_df)
            builder = GraphBuilder(processed_tx, acc_df)
            client_graphs[client_id] = builder.build()
            
        return client_graphs, preprocessor

if __name__ == "__main__":
    loader = DataLoader()
    graphs, prep = loader.get_federated_graphs(non_iid_degree="moderate")
    for k, v in graphs.items():
        print(k, v)
