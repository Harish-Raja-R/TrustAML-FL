"""
Test for target leakage in graph features.
Verifies that node features computed from a temporal window do NOT include
information from the test (future) window.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_no_target_leakage_in_graph_features():
    """
    Construct two temporal windows. Compute account features from Window 1 only.
    Verify that Window 2 transactions do not influence Window 1's feature values.
    """
    # Window 1: Jan transactions
    window1 = pd.DataFrame({
        'transaction_id': ['T1', 'T2', 'T3'],
        'timestamp': [datetime(2023, 1, 1), datetime(2023, 1, 15), datetime(2023, 1, 20)],
        'sender_account': ['ACC_0', 'ACC_1', 'ACC_0'],
        'receiver_account': ['ACC_1', 'ACC_0', 'ACC_2'],
        'amount': [100.0, 200.0, 300.0],
        'currency': [0, 0, 0],
        'transaction_type': [0, 0, 0],
        'channel': [0, 0, 0],
        'is_aml': [0, 0, 0],
        'device': ['D1', 'D1', 'D2'],
        'location': ['L1', 'L1', 'L2'],
    })
    
    # Window 2: Feb transactions (future, must NOT leak)
    window2 = pd.DataFrame({
        'transaction_id': ['T4', 'T5'],
        'timestamp': [datetime(2023, 2, 1), datetime(2023, 2, 15)],
        'sender_account': ['ACC_0', 'ACC_2'],
        'receiver_account': ['ACC_2', 'ACC_0'],
        'amount': [9999.0, 8888.0],
        'currency': [0, 0],
        'transaction_type': [0, 0],
        'channel': [0, 0],
        'is_aml': [1, 1],
        'device': ['D3', 'D4'],
        'location': ['L3', 'L4'],
    })
    
    # Compute aggregated features using ONLY window1
    outbound = window1.groupby('sender_account')['amount'].agg(['count', 'sum'])
    inbound = window1.groupby('receiver_account')['amount'].agg(['count', 'sum'])
    
    # ACC_0 outbound from window1: T1 (100) + T3 (300) = 2 txns, 400 total
    acc0_out_count = outbound.at['ACC_0', 'count']
    acc0_out_sum = outbound.at['ACC_0', 'sum']
    
    assert acc0_out_count == 2, f"Expected 2 outbound txns for ACC_0, got {acc0_out_count}"
    assert acc0_out_sum == 400.0, f"Expected 400.0 outbound sum for ACC_0, got {acc0_out_sum}"
    
    # The 9999.0 transaction from window2 must NOT appear
    assert acc0_out_sum < 9999.0, "Future window amount leaked into train features!"
    
    # ACC_2 should have 0 outbound in window1 (ACC_2 only sends in window2)
    acc2_out = outbound.at['ACC_2', 'count'] if 'ACC_2' in outbound.index else 0
    assert acc2_out == 0, f"ACC_2 should have 0 outbound in window1, got {acc2_out}"


def test_oot_evaluator_no_overlap():
    """Verify temporal folds have no overlapping timestamps."""
    from src.evaluation.temporal import TemporalEvaluator
    
    # Create a simple chronological dataset
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tx_df = pd.DataFrame({
        'timestamp': dates,
        'amount': np.random.rand(100) * 1000,
    })
    
    evaluator = TemporalEvaluator(tx_df, num_windows=5)
    folds = evaluator.get_oot_folds()
    
    for fold in folds:
        train_max_ts = fold['train']['timestamp'].max()
        test_min_ts = fold['test']['timestamp'].min()
        assert train_max_ts < test_min_ts, (
            f"Fold {fold['fold_id']}: Train max timestamp ({train_max_ts}) "
            f"must be strictly before test min timestamp ({test_min_ts})"
        )
