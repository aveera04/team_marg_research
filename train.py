"""
train.py
--------
Orchestrates training of all DARS-GNN sub-models and computes the final
Digital Arrest Risk Score (DARS):

    DARS = α·BRS + β·GRS + γ·SNS + δ·VA

Also handles:
  - Model serialization (pickle / PyTorch / JSON)
  - Graph edge-index construction for PyG
  - SNS propagation from node-level back to transaction-level
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.utils import to_undirected
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ---------------------------------------------------------------------------
# DARS Score Combination
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "alpha": 0.40,   # Behavioral Risk Score
    "beta":  0.15,   # Geographic Risk Score
    "gamma": 0.35,   # Scam Network Score
    "delta": 0.10,   # Vulnerability Assessment
}


def combine_scores(
    brs: np.ndarray,
    grs: np.ndarray,
    sns: np.ndarray,
    va:  np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Compute the Digital Arrest Risk Score (DARS) as a weighted sum:

        DARS = α·BRS_norm + β·GRS_norm + γ·SNS_norm + δ·VA_norm

    All inputs are normalized to [0, 1] internally before combining.
    The result is scaled to [0, 100].

    Parameters
    ----------
    brs, grs, sns, va : np.ndarray  (same length)
        Component scores.
    weights : dict with keys 'alpha', 'beta', 'gamma', 'delta' (sum to 1).
              Defaults to DEFAULT_WEIGHTS if None.

    Returns
    -------
    np.ndarray of DARS values in [0, 100].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    def _norm(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    brs_n = _norm(brs)
    grs_n = _norm(grs)
    sns_n = _norm(sns)
    va_n  = _norm(va)

    dars = (
        weights["alpha"] * brs_n +
        weights["beta"]  * grs_n +
        weights["gamma"] * sns_n +
        weights["delta"] * va_n
    )

    dars = np.clip(dars * 100.0, 0.0, 100.0)
    return dars.astype(np.float32)


# ---------------------------------------------------------------------------
# Build PyG Edge Index
# ---------------------------------------------------------------------------

def build_edge_index(
    edges: list[tuple[str, str]],
    node_ids: list[str],
    undirected: bool = False,
) -> "torch.Tensor":
    """
    Convert a list of (src, dst) string edges into a PyTorch Geometric
    edge_index tensor of shape (2, n_edges).

    Parameters
    ----------
    edges      : List of (source_account_id, dest_account_id) tuples.
    node_ids   : Ordered list of all node IDs (row indices in feature matrix).
    undirected : If True, add reverse edges for each directed edge.

    Returns
    -------
    torch.Tensor of shape (2, n_edges), dtype=torch.long.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for build_edge_index().")

    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    src_list, dst_list = [], []
    skipped = 0

    for (s, d) in edges:
        si = id2idx.get(s)
        di = id2idx.get(d)
        if si is None or di is None:
            skipped += 1
            continue
        src_list.append(si)
        dst_list.append(di)

    if skipped:
        logger.warning(f"Skipped {skipped} edges with unknown node IDs.")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    if undirected and PYG_AVAILABLE:
        edge_index = to_undirected(edge_index)

    logger.info(f"Edge index built: {edge_index.shape[1]} edges, {len(node_ids)} nodes.")
    return edge_index


# ---------------------------------------------------------------------------
# Map node-level SNS back to transaction-level
# ---------------------------------------------------------------------------

def map_sns_to_transactions(
    upi_df: pd.DataFrame,
    node_ids: list[str],
    sns_scores: np.ndarray,
    account_col: str = "sender_bank",
) -> np.ndarray:
    """
    Each UPI transaction row is mapped to a sender node's SNS score.

    Since UPI data uses bank names rather than exact account IDs, we match
    on `account_col` (default: 'sender_bank'). If no match, score = median SNS.

    Parameters
    ----------
    upi_df      : UPI DataFrame
    node_ids    : Ordered list of account IDs (from build_node_feature_matrix)
    sns_scores  : np.ndarray of SNS values (same order as node_ids)
    account_col : Column in upi_df to join on

    Returns
    -------
    np.ndarray of SNS values per transaction row, shape (n_transactions,).
    """
    id2sns = {nid: float(s) for nid, s in zip(node_ids, sns_scores)}
    default_sns = float(np.median(sns_scores)) if len(sns_scores) > 0 else 0.0

    if account_col in upi_df.columns:
        txn_sns = upi_df[account_col].map(id2sns).fillna(default_sns).values
    else:
        logger.warning(
            f"Column '{account_col}' not in UPI data. Assigning default SNS to all transactions."
        )
        txn_sns = np.full(len(upi_df), default_sns, dtype=np.float32)

    return txn_sns.astype(np.float32)


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_models(
    iso_model=None,
    gat_model=None,
    xgb_model=None,
    output_dir: str = ".",
):
    """
    Persist trained models to disk.

    Files saved:
      - iso_forest_model.pkl    (IsolationForest)
      - gat_model.pth           (GAT state dict)
      - xgb_shell_model.json    (XGBoost)

    Parameters
    ----------
    iso_model  : BehavioralAnomalyModel instance
    gat_model  : GATNetwork instance
    xgb_model  : ShellAccountModel instance
    output_dir : Directory to save into (default: current directory)
    """
    os.makedirs(output_dir, exist_ok=True)

    if iso_model is not None:
        path = os.path.join(output_dir, "iso_forest_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(iso_model, f)
        logger.info(f"IsolationForest saved to {path}")

    if gat_model is not None and TORCH_AVAILABLE:
        path = os.path.join(output_dir, "gat_model.pth")
        torch.save(gat_model.state_dict(), path)
        logger.info(f"GAT model saved to {path}")

    if xgb_model is not None:
        path = os.path.join(output_dir, "xgb_shell_model.json")
        xgb_model.clf.save_model(path)
        logger.info(f"XGBoost model saved to {path}")


def load_iso_model(path: str):
    """Load a pickled BehavioralAnomalyModel."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gat_model(path: str, model_class, **kwargs):
    """Load GAT weights into a GATNetwork instance."""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_xgb_model(path: str, model_class):
    """Load XGBoost model from JSON."""
    inst = model_class.__new__(model_class)
    import xgboost as xgb
    inst.clf = xgb.XGBClassifier()
    inst.clf.load_model(path)
    inst._fitted = True
    return inst
