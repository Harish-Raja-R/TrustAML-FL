import pandas as pd
import numpy as np
import torch
try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("WARNING: torch_geometric not found. Fallback mode enabled.")

class GraphBuilder:
    def __init__(self, tx_df, acc_df):
        self.tx_df = tx_df
        self.acc_df = acc_df
        
        # Mappings
        self.node_mapping = {
            'account': {},
            'transaction': {},
            'device': {},
            'location': {}
        }
        
    def _create_mapping(self, entities, node_type):
        mapping = {idx: i for i, idx in enumerate(entities.unique())}
        self.node_mapping[node_type] = mapping
        return mapping
        
    def build(self):
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required to build the HeteroData graph.")
            
        data = HeteroData()
        
        # Extract unique entities
        accounts = pd.concat([self.tx_df['sender_account'], self.tx_df['receiver_account']]).dropna().unique()
        transactions = self.tx_df['transaction_id'].unique()
        devices = self.tx_df['device'].dropna().unique()
        locations = self.tx_df['location'].dropna().unique()
        
        # Create mappings
        acc_map = self._create_mapping(pd.Series(accounts), 'account')
        tx_map = self._create_mapping(pd.Series(transactions), 'transaction')
        dev_map = self._create_mapping(pd.Series(devices), 'device')
        loc_map = self._create_mapping(pd.Series(locations), 'location')
        
        # Add nodes
        data['account'].num_nodes = len(accounts)
        data['transaction'].num_nodes = len(transactions)
        data['device'].num_nodes = len(devices)
        data['location'].num_nodes = len(locations)
        
        # Transaction node features (amount, currency, type, channel)
        # Assuming preprocessing has already encoded categorical and scaled numerical features
        tx_features = []
        tx_labels = []
        tx_times = []
        
        for _, row in self.tx_df.iterrows():
            idx = tx_map[row['transaction_id']]
            # amount, currency, transaction_type, channel
            feat = [row['amount'], row['currency'], row['transaction_type'], row['channel']]
            tx_features.append(feat)
            tx_labels.append(row['is_aml'])
            # Convert timestamp to int for temporal sorting
            if isinstance(row['timestamp'], str):
                ts = pd.to_datetime(row['timestamp']).timestamp()
            else:
                ts = row['timestamp'].timestamp()
            tx_times.append(ts)
            
        data['transaction'].x = torch.tensor(tx_features, dtype=torch.float)
        data['transaction'].y = torch.tensor(tx_labels, dtype=torch.long)
        data['transaction'].timestamp = torch.tensor(tx_times, dtype=torch.float)
        
        # Statistically realistic node features computed ONLY from the given temporal window (self.tx_df)
        # Avoids target leakage by restricting calculations to the visible graph.
        
        # 1. Account Features: [transaction_count, inbound_count, outbound_count, total_inbound_amount, total_outbound_amount]
        account_features = []
        # Precompute aggregations
        outbound_stats = self.tx_df.groupby('sender_account')['amount'].agg(['count', 'sum']).fillna(0)
        inbound_stats = self.tx_df.groupby('receiver_account')['amount'].agg(['count', 'sum']).fillna(0)
        
        for acc in accounts:
            out_count = outbound_stats.at[acc, 'count'] if acc in outbound_stats.index else 0
            out_sum = outbound_stats.at[acc, 'sum'] if acc in outbound_stats.index else 0
            in_count = inbound_stats.at[acc, 'count'] if acc in inbound_stats.index else 0
            in_sum = inbound_stats.at[acc, 'sum'] if acc in inbound_stats.index else 0
            total_count = out_count + in_count
            account_features.append([total_count, in_count, out_count, in_sum, out_sum])
            
        data['account'].x = torch.tensor(account_features, dtype=torch.float)
        
        # 2. Device Features: [transaction_count, total_amount]
        device_features = []
        dev_stats = self.tx_df.groupby('device')['amount'].agg(['count', 'sum']).fillna(0)
        for dev in devices:
            count = dev_stats.at[dev, 'count'] if dev in dev_stats.index else 0
            tsum = dev_stats.at[dev, 'sum'] if dev in dev_stats.index else 0
            device_features.append([count, tsum])
            
        data['device'].x = torch.tensor(device_features, dtype=torch.float)
        
        # 3. Location Features: [transaction_count, total_amount]
        location_features = []
        loc_stats = self.tx_df.groupby('location')['amount'].agg(['count', 'sum']).fillna(0)
        for loc in locations:
            count = loc_stats.at[loc, 'count'] if loc in loc_stats.index else 0
            tsum = loc_stats.at[loc, 'sum'] if loc in loc_stats.index else 0
            location_features.append([count, tsum])
            
        data['location'].x = torch.tensor(location_features, dtype=torch.float)
        
        # Add edges
        # account -> SENDS -> transaction
        senders = self.tx_df['sender_account'].map(acc_map).values
        txs = self.tx_df['transaction_id'].map(tx_map).values
        
        valid_sends = ~pd.isna(senders)
        send_edge_index = torch.tensor(np.array([senders[valid_sends], txs[valid_sends]]), dtype=torch.long)
        data['account', 'sends', 'transaction'].edge_index = send_edge_index
        
        # transaction -> RECEIVES -> account
        receivers = self.tx_df['receiver_account'].map(acc_map).values
        valid_recvs = ~pd.isna(receivers)
        recv_edge_index = torch.tensor(np.array([txs[valid_recvs], receivers[valid_recvs]]), dtype=torch.long)
        data['transaction', 'receives', 'account'].edge_index = recv_edge_index
        
        # transaction -> USES -> device
        devs = self.tx_df['device'].map(dev_map).values
        valid_devs = ~pd.isna(devs)
        dev_edge_index = torch.tensor(np.array([txs[valid_devs], devs[valid_devs]]), dtype=torch.long)
        data['transaction', 'uses', 'device'].edge_index = dev_edge_index
        
        # transaction -> LOCATED_AT -> location
        locs = self.tx_df['location'].map(loc_map).values
        valid_locs = ~pd.isna(locs)
        loc_edge_index = torch.tensor(np.array([txs[valid_locs], locs[valid_locs]]), dtype=torch.long)
        data['transaction', 'located_at', 'location'].edge_index = loc_edge_index
        
        # Add reverse edges to make the graph undirected for GNN message passing
        if hasattr(data, 'to_undirected'):
            # PyG native
            data = data.to_undirected()
            
        return data

if __name__ == "__main__":
    from src.data.preprocessing import DataPreprocessor
    tx_df = pd.read_csv("data/synthetic/transactions.csv")
    acc_df = pd.read_csv("data/synthetic/accounts.csv")
    
    preprocessor = DataPreprocessor()
    tx_df = preprocessor.fit_transform(tx_df, is_train=True)
    
    builder = GraphBuilder(tx_df, acc_df)
    hetero_data = builder.build()
    print("Graph built successfully:")
    print(hetero_data)
