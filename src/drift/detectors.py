import numpy as np
from scipy.stats import wasserstein_distance, entropy

def compute_kl_divergence(p, q, bins=10):
    """
    Compute KL divergence between two 1D distributions.
    """
    hist_p, bin_edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid div by zero
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    
    hist_p /= np.sum(hist_p)
    hist_q /= np.sum(hist_q)
    
    return entropy(hist_p, hist_q)

def detect_feature_drift(ref_features, new_features):
    """
    Detect drift on feature level.
    """
    drifts = {}
    for i in range(ref_features.shape[1]):
        wd = wasserstein_distance(ref_features[:, i], new_features[:, i])
        kl = compute_kl_divergence(ref_features[:, i], new_features[:, i])
        drifts[f"feature_{i}"] = {"wasserstein": float(wd), "kl": float(kl)}
    return drifts
