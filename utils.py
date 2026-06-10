"""
utils.py
--------
Utility helpers and lightweight unit tests for the DARS-GNN pipeline.

Run tests with:
    python utils.py
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Normalization helpers
# =============================================================================

def normalize_to_range(arr: np.ndarray, low: float = 0.0, high: float = 100.0) -> np.ndarray:
    """Min-max normalize array to [low, high]."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, (low + high) / 2.0)
    return low + (arr - mn) / (mx - mn) * (high - low)


def threshold_flag(scores: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    """Return binary flags where score >= threshold."""
    return (np.asarray(scores) >= threshold).astype(int)


# =============================================================================
# Unit tests
# =============================================================================

def test_resample_smote():
    from features import resample_smote
    X = np.random.rand(200, 5)
    y = np.array([0] * 190 + [1] * 10)
    X_res, y_res = resample_smote(X, y)
    assert len(y_res) > len(y), "SMOTE should add samples"
    assert sum(y_res == 1) > 10, "SMOTE should oversample minority class"
    print("✓ test_resample_smote passed")


def test_combine_scores_range():
    from train import combine_scores
    n = 500
    brs = np.random.uniform(0, 100, n)
    grs = np.random.uniform(0, 100, n)
    sns = np.random.uniform(0, 100, n)
    va  = np.random.uniform(0, 1, n)
    dars = combine_scores(brs, grs, sns, va)
    assert dars.min() >= 0.0 and dars.max() <= 100.0, "DARS must be in [0,100]"
    assert len(dars) == n
    print("✓ test_combine_scores_range passed")


def test_normalize_to_range():
    arr = np.array([0.0, 50.0, 100.0])
    out = normalize_to_range(arr, 0, 100)
    assert abs(out[0] - 0.0) < 1e-6 and abs(out[-1] - 100.0) < 1e-6
    print("✓ test_normalize_to_range passed")


def test_threshold_flag():
    scores = np.array([30.0, 60.0, 75.0, 59.9])
    flags  = threshold_flag(scores, threshold=60.0)
    assert list(flags) == [0, 1, 1, 0]
    print("✓ test_threshold_flag passed")


def test_behavioral_features():
    """Smoke-test behavioral feature computation."""
    import pandas as pd
    from features import compute_behavioral_features
    df = pd.DataFrame({
        "sender_bank":      ["HDFC", "HDFC", "SBI", "SBI"],
        "sender_age_group": ["26-35", "26-35", "46-55", "46-55"],
        "amount_inr":       [1000.0, 50000.0, 500.0, 600.0],
        "receiver_bank":    ["SBI", "ICICI", "HDFC", "HDFC"],
        "hour_of_day":      [10, 10, 14, 14],
    })
    df = compute_behavioral_features(df)
    assert "high_amount_flag" in df.columns
    assert "tx_count" in df.columns
    assert df.loc[1, "high_amount_flag"] == 1, "50000 should be high_amount vs mean 25500"
    print("✓ test_behavioral_features passed")


def test_geographic_risk():
    import pandas as pd
    from features import compute_geographic_risk
    df = pd.DataFrame({
        "sender_bank":      ["HDFC", "HDFC"],
        "sender_age_group": ["26-35", "26-35"],
        "sender_state":     ["Maharashtra", "Bihar"],
    })
    df = compute_geographic_risk(df)
    assert "GRS" in df.columns
    # Bihar is a high-risk state; Maharashtra is not
    assert df.loc[1, "GRS"] >= df.loc[0, "GRS"]
    print("✓ test_geographic_risk passed")


def test_iso_forest_model():
    from models import BehavioralAnomalyModel
    X = np.random.rand(300, 5)
    model = BehavioralAnomalyModel(contamination=0.05)
    model.fit(X)
    scores = model.score(X)
    assert scores.shape == (300,)
    assert 0.0 <= scores.min() and scores.max() <= 100.0
    print("✓ test_iso_forest_model passed")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("\n── Running DARS-GNN unit tests ──")
    test_normalize_to_range()
    test_threshold_flag()
    test_combine_scores_range()
    test_behavioral_features()
    test_geographic_risk()
    test_iso_forest_model()
    test_resample_smote()
    print("\n✅ All tests passed!\n")
