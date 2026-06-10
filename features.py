"""
features.py
-----------
Feature engineering for the DARS-GNN pipeline:
  - Behavioral Features  (BF)
  - Geographic Risk Score (GRS)
  - Vulnerability Assessment (VA)
  - Categorical Encoding
  - SMOTE Resampling for class imbalance
  - Graph Node Feature Matrix construction
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Behavioral Features
# ---------------------------------------------------------------------------

def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-behavior aggregate features grouped by sender identity.

    Features added per row:
      - tx_count        : total transactions from this sender group
      - avg_amt         : historical mean transaction amount
      - std_amt         : historical std deviation of transaction amount
      - uniq_rec        : count of unique receivers used
      - high_amount_flag: 1 if current amount > 3× historical mean

    Parameters
    ----------
    df : pd.DataFrame
        UPI transaction DataFrame (output of load_upi_data).

    Returns
    -------
    pd.DataFrame
        DataFrame with added behavioral feature columns.
    """
    # Identify grouping key columns
    grp_keys = [c for c in ["sender_bank", "sender_age_group"] if c in df.columns]
    amt_col  = _get_col(df, ["amount_inr", "amount"])
    rec_col  = _get_col(df, ["receiver_bank", "receiver_id", "receiver"])

    if not grp_keys or not amt_col:
        logger.warning("Behavioral features skipped: missing grouping or amount column.")
        df["tx_count"] = 1
        df["avg_amt"]  = df.get(amt_col, pd.Series(0.0, index=df.index))
        df["std_amt"]  = 0.0
        df["uniq_rec"] = 1
        df["high_amount_flag"] = 0
        return df

    agg_dict = {
        "tx_count":   (amt_col, "count"),
        "avg_amt":    (amt_col, "mean"),
        "std_amt":    (amt_col, "std"),
    }
    if rec_col:
        agg_dict["uniq_rec"] = (rec_col, "nunique")

    agg = df.groupby(grp_keys).agg(**{k: v for k, v in agg_dict.items()}).reset_index()
    agg["std_amt"] = agg["std_amt"].fillna(0.0)
    if "uniq_rec" not in agg.columns:
        agg["uniq_rec"] = 1

    df = df.merge(agg, on=grp_keys, how="left")

    df["high_amount_flag"] = (df[amt_col] > 3 * df["avg_amt"].fillna(0)).astype(int)

    # Velocity anomaly: transactions in the same hour spike
    if "hour_of_day" in df.columns:
        hour_grp = grp_keys + ["hour_of_day"]
        hour_cnt = (
            df.groupby(hour_grp).size().reset_index(name="hourly_tx_count")
        )
        df = df.merge(hour_cnt, on=hour_grp, how="left")
    else:
        df["hourly_tx_count"] = 1

    logger.info("Behavioral features computed.")
    return df


# ---------------------------------------------------------------------------
# Geographic Risk Score
# ---------------------------------------------------------------------------

# Known high-risk state abbreviations / names (heuristic)
HIGH_RISK_STATES = {
    "jharkhand", "bihar", "uttar pradesh", "up", "rajasthan",
    "haryana", "west bengal", "wb",
}


def compute_geographic_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Geographic Risk Score (GRS) for each transaction.

    Logic:
      1. Determine each sender group's historical set of states.
      2. If the current transaction state is outside that set → high risk (80).
      3. If the state is in a known high-risk list                → elevated (60).
      4. Otherwise                                               → low risk (10).

    GRS values are in [0, 100].

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'GRS' column.
    """
    grp_keys  = [c for c in ["sender_bank", "sender_age_group"] if c in df.columns]
    state_col = _get_col(df, ["sender_state", "state", "location"])

    if not state_col:
        logger.warning("Geographic risk skipped: no state column found. GRS set to 0.")
        df["GRS"] = 0.0
        return df

    if grp_keys:
        hist = (
            df.groupby(grp_keys)[state_col]
            .agg(lambda s: set(s.dropna().str.lower()))
            .reset_index()
            .rename(columns={state_col: "_state_set"})
        )
        df = df.merge(hist, on=grp_keys, how="left")

        def _grs(row):
            cur_state = str(row[state_col]).lower()
            hist_states: set = row.get("_state_set", set())
            if cur_state not in hist_states:
                return 80.0
            if cur_state in HIGH_RISK_STATES:
                return 60.0
            return 10.0

        df["GRS"] = df.apply(_grs, axis=1)
        df.drop(columns=["_state_set"], inplace=True, errors="ignore")
    else:
        df["GRS"] = df[state_col].str.lower().apply(
            lambda s: 60.0 if s in HIGH_RISK_STATES else 10.0
        )

    logger.info("Geographic risk scores computed.")
    return df


# ---------------------------------------------------------------------------
# Vulnerability Assessment
# ---------------------------------------------------------------------------

AGE_VULNERABILITY_MAP = {
    "18-25": 0.25,
    "26-35": 0.10,
    "36-45": 0.15,
    "46-55": 0.25,
    "56+":   0.40,
    "60+":   0.45,
}

DEVICE_VULNERABILITY_MAP = {
    "android": 0.10,
    "ios":     0.05,
    "unknown": 0.20,
}


def compute_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Vulnerability Assessment (VA) score for each transaction sender.

    Combines:
      - Age-group based risk (older = higher risk)
      - Device type risk (unknown device = higher risk)

    VA is in [0, 1].

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'VA' column.
    """
    age_col    = _get_col(df, ["sender_age_group", "age_group"])
    device_col = _get_col(df, ["device_type", "device"])

    va = pd.Series(0.15, index=df.index)  # default

    if age_col:
        age_va = df[age_col].map(AGE_VULNERABILITY_MAP).fillna(0.15)
        va = va + age_va

    if device_col:
        dev_va = df[device_col].str.lower().map(DEVICE_VULNERABILITY_MAP).fillna(0.15)
        va = va + dev_va

    # Normalize to [0,1]
    va = (va - va.min()) / (va.max() - va.min() + 1e-9)
    df["VA"] = va

    logger.info("Vulnerability scores computed.")
    return df


# ---------------------------------------------------------------------------
# Categorical Encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns in-place.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list of column names to encode (auto-detected if None).

    Returns
    -------
    df       : pd.DataFrame with new '<col>_enc' columns
    encoders : dict mapping column name -> fitted LabelEncoder
    """
    if columns is None:
        # Auto-detect: object/category columns
        columns = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in ["transaction_id", "day_of_week"]
        ]

    encoders: dict = {}
    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        encoders[col] = le
        logger.debug(f"Encoded column: {col} ({len(le.classes_)} classes)")

    logger.info(f"Categorical encoding complete for {len(encoders)} columns.")
    return df, encoders


# ---------------------------------------------------------------------------
# SMOTE Resampling
# ---------------------------------------------------------------------------

def resample_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the minority class.

    Parameters
    ----------
    X             : Feature matrix (n_samples, n_features)
    y             : Labels (n_samples,)
    random_state  : Random seed.
    k_neighbors   : SMOTE k_neighbors parameter.

    Returns
    -------
    X_res, y_res : Resampled arrays.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Class distribution before SMOTE: {class_dist}")

    # Only apply SMOTE if minority class is present
    if len(unique) < 2 or min(counts) < k_neighbors + 1:
        logger.warning("SMOTE skipped: insufficient minority samples.")
        return X, y

    sm = SMOTE(random_state=random_state, k_neighbors=min(k_neighbors, min(counts) - 1))
    X_res, y_res = sm.fit_resample(X, y)

    unique2, counts2 = np.unique(y_res, return_counts=True)
    logger.info(f"Class distribution after SMOTE: {dict(zip(unique2, counts2))}")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Graph Node Feature Matrix
# ---------------------------------------------------------------------------

def build_node_feature_matrix(
    node_features: dict,
    feature_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Convert the node_features dict (from construct_transaction_graph) into a
    numpy feature matrix suitable for PyTorch Geometric.

    Parameters
    ----------
    node_features : dict  {account_id -> {feature_name -> value}}
    feature_keys  : list of feature names to use (auto-detected if None)

    Returns
    -------
    X        : np.ndarray of shape (n_nodes, n_features)
    node_ids : list of account IDs in row order
    feat2idx : dict mapping feature name -> column index
    """
    node_ids = list(node_features.keys())

    if feature_keys is None:
        # Collect all keys present across nodes
        all_keys: set = set()
        for feats in node_features.values():
            all_keys.update(feats.keys())
        feature_keys = sorted(all_keys)

    feat2idx = {k: i for i, k in enumerate(feature_keys)}
    X = np.zeros((len(node_ids), len(feature_keys)), dtype=np.float32)

    for row_idx, nid in enumerate(node_ids):
        for k, col_idx in feat2idx.items():
            X[row_idx, col_idx] = node_features[nid].get(k, 0.0)

    # Normalize each feature column to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    logger.info(
        f"Node feature matrix built: {X.shape[0]} nodes × {X.shape[1]} features."
    )
    return X, node_ids, feat2idx


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate column name found in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None
