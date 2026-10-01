import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random

class AMLDataGenerator:
    def __init__(self, num_banks=3, num_customers=1000, num_transactions=10000, start_date="2023-01-01", seed=42):
        self.num_banks = num_banks
        self.num_customers = num_customers
        self.num_transactions = num_transactions
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Data containers
        self.customers = []
        self.accounts = []
        self.merchants = []
        self.devices = []
        self.locations = []
        self.transactions = []

    def generate_entities(self):
        print(f"Generating {self.num_customers} customers and associated entities...")
        # Banks
        bank_ids = [f"BANK_{i}" for i in range(1, self.num_banks + 1)]
        
        # Customers & Accounts
        for i in range(self.num_customers):
            cust_id = f"CUST_{i}"
            self.customers.append({"customer_id": cust_id, "risk_score": np.random.beta(2, 5)})
            
            # Each customer has 1-3 accounts
            num_accs = np.random.randint(1, 4)
            for j in range(num_accs):
                acc_id = f"ACC_{i}_{j}"
                bank = np.random.choice(bank_ids)
                self.accounts.append({"account_id": acc_id, "customer_id": cust_id, "bank_id": bank})

        # Merchants
        num_merchants = max(10, self.num_customers // 10)
        for i in range(num_merchants):
            self.merchants.append({"merchant_id": f"MERCH_{i}", "category": np.random.choice(["retail", "travel", "crypto", "services", "digital"])})
            
        # Devices
        for i in range(self.num_customers * 2):
            self.devices.append({"device_id": f"DEV_{i}", "type": np.random.choice(["mobile", "desktop", "tablet"])})
            
        # Locations
        countries = ["US", "UK", "CA", "SG", "UAE", "CH", "KY"] # KY = Cayman, high risk dummy
        for i in range(self.num_customers):
            self.locations.append({"location_id": f"LOC_{i}", "country": np.random.choice(countries, p=[0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])})

    def generate_normal_transactions(self):
        print(f"Generating normal transactions...")
        types = ["transfer", "payment", "withdrawal", "deposit"]
        channels = ["online", "mobile", "branch", "atm"]
        currencies = ["USD", "EUR", "GBP"]
        
        acc_ids = [a["account_id"] for a in self.accounts]
        merch_ids = [m["merchant_id"] for m in self.merchants]
        dev_ids = [d["device_id"] for d in self.devices]
        loc_ids = [l["location_id"] for l in self.locations]
        
        for i in range(self.num_transactions):
            sender = np.random.choice(acc_ids)
            receiver = np.random.choice(acc_ids) if np.random.random() > 0.3 else None
            merchant = np.random.choice(merch_ids) if receiver is None else None
            
            amount = np.random.lognormal(mean=4, sigma=1.5) # mostly small, some large
            
            tx = {
                "transaction_id": f"TXN_{uuid.uuid4().hex[:8]}",
                "timestamp": self.start_date + timedelta(minutes=np.random.randint(0, 365*24*60)),
                "sender_account": sender,
                "receiver_account": receiver,
                "amount": round(amount, 2),
                "currency": np.random.choice(currencies, p=[0.8, 0.15, 0.05]),
                "merchant": merchant,
                "device": np.random.choice(dev_ids),
                "location": np.random.choice(loc_ids),
                "transaction_type": np.random.choice(types),
                "channel": np.random.choice(channels),
                "is_fraud": 0,
                "is_aml": 0,
                "pattern": "normal"
            }
            self.transactions.append(tx)

    def inject_aml_patterns(self):
        print("Injecting AML patterns...")
        acc_ids = [a["account_id"] for a in self.accounts]
        dev_ids = [d["device_id"] for d in self.devices]
        loc_ids = [l["location_id"] for l in self.locations]
        
        # 1. Structuring (Smurfing): Multiple small deposits below reporting threshold
        self._inject_structuring(acc_ids, dev_ids, loc_ids)
        # 2. Circular transactions: A -> B -> C -> A
        self._inject_circular(acc_ids, dev_ids, loc_ids)
        # 3. Fan-out (Mule behavior): One account sends to many
        self._inject_fan_out(acc_ids, dev_ids, loc_ids)
        # 4. Cross-border chains
        self._inject_cross_border(acc_ids, dev_ids, loc_ids)
        
    def _create_base_aml_txn(self, sender, receiver, amount, dev_ids, loc_ids, time_offset, pattern_name):
        return {
            "transaction_id": f"AML_{uuid.uuid4().hex[:8]}",
            "timestamp": self.start_date + timedelta(minutes=time_offset),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": round(amount, 2),
            "currency": "USD",
            "merchant": None,
            "device": np.random.choice(dev_ids),
            "location": np.random.choice(loc_ids),
            "transaction_type": "transfer",
            "channel": "online",
            "is_fraud": 1,
            "is_aml": 1,
            "pattern": pattern_name
        }

    def _inject_structuring(self, acc_ids, dev_ids, loc_ids, num_instances=20):
        for _ in range(num_instances):
            receiver = np.random.choice(acc_ids)
            base_time = np.random.randint(0, 360*24*60)
            for _ in range(np.random.randint(5, 10)):
                sender = np.random.choice(acc_ids)
                amount = np.random.uniform(9000, 9999) # Below $10k threshold
                tx = self._create_base_aml_txn(sender, receiver, amount, dev_ids, loc_ids, base_time + np.random.randint(1, 60), "structuring")
                self.transactions.append(tx)

    def _inject_circular(self, acc_ids, dev_ids, loc_ids, num_instances=15):
        for _ in range(num_instances):
            cycle_nodes = np.random.choice(acc_ids, size=np.random.randint(3, 6), replace=False)
            base_time = np.random.randint(0, 360*24*60)
            amount = np.random.uniform(50000, 500000)
            for k in range(len(cycle_nodes)):
                sender = cycle_nodes[k]
                receiver = cycle_nodes[(k + 1) % len(cycle_nodes)]
                amt = amount * np.random.uniform(0.95, 1.0) # slight deduction
                tx = self._create_base_aml_txn(sender, receiver, amt, dev_ids, loc_ids, base_time + k*10, "circular")
                self.transactions.append(tx)

    def _inject_fan_out(self, acc_ids, dev_ids, loc_ids, num_instances=20):
        for _ in range(num_instances):
            sender = np.random.choice(acc_ids)
            base_time = np.random.randint(0, 360*24*60)
            for _ in range(np.random.randint(10, 30)):
                receiver = np.random.choice(acc_ids)
                amount = np.random.uniform(1000, 5000)
                tx = self._create_base_aml_txn(sender, receiver, amount, dev_ids, loc_ids, base_time + np.random.randint(1, 120), "fan_out")
                self.transactions.append(tx)
                
    def _inject_cross_border(self, acc_ids, dev_ids, loc_ids, num_instances=15):
        # High value transfer through multiple nodes to an offshore location
        high_risk_locs = [l for l in self.locations if l['country'] in ['KY', 'UAE', 'SG']]
        if not high_risk_locs:
            high_risk_locs = self.locations
        
        for _ in range(num_instances):
            chain_nodes = np.random.choice(acc_ids, size=4, replace=False)
            base_time = np.random.randint(0, 360*24*60)
            amount = np.random.uniform(100000, 1000000)
            
            for k in range(len(chain_nodes) - 1):
                sender = chain_nodes[k]
                receiver = chain_nodes[k+1]
                loc = np.random.choice(loc_ids) if k < len(chain_nodes)-2 else np.random.choice([l['location_id'] for l in high_risk_locs])
                
                tx = self._create_base_aml_txn(sender, receiver, amount, dev_ids, [loc], base_time + k*60, "cross_border")
                amount = amount * np.random.uniform(0.9, 0.99)
                self.transactions.append(tx)

    def generate(self):
        self.generate_entities()
        self.generate_normal_transactions()
        self.inject_aml_patterns()
        
        # Sort by timestamp
        self.transactions.sort(key=lambda x: x["timestamp"])
        
        df_tx = pd.DataFrame(self.transactions)
        df_acc = pd.DataFrame(self.accounts)
        
        # Map bank_id to transactions via sender_account
        df_tx = df_tx.merge(df_acc[['account_id', 'bank_id']], left_on='sender_account', right_on='account_id', how='left')
        df_tx.rename(columns={'bank_id': 'sender_bank'}, inplace=True)
        df_tx.drop('account_id', axis=1, inplace=True)
        
        df_tx = df_tx.merge(df_acc[['account_id', 'bank_id']], left_on='receiver_account', right_on='account_id', how='left')
        df_tx.rename(columns={'bank_id': 'receiver_bank'}, inplace=True)
        df_tx.drop('account_id', axis=1, inplace=True)
        
        print(f"Generated {len(df_tx)} total transactions. AML cases: {df_tx['is_aml'].sum()}")
        return df_tx, df_acc

if __name__ == "__main__":
    generator = AMLDataGenerator(num_banks=3, num_customers=5000, num_transactions=20000)
    df_tx, df_acc = generator.generate()
    df_tx.to_csv("data/synthetic/transactions.csv", index=False)
    df_acc.to_csv("data/synthetic/accounts.csv", index=False)
    print("Data saved to data/synthetic/")
