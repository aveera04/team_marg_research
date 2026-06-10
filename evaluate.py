"""
evaluate.py
-----------
Evaluation and visualization for the DARS-GNN pipeline:

  - Classification metrics (confusion matrix, precision/recall/F1, AUC)
  - Risk score distribution plot
  - SHAP feature importance plot (XGBoost shell model)
  - Transaction network visualization (NetworkX)
  - Fraud vs. legitimate comparison boxplot
  - Save all metrics to JSON
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional imports ────────────────────────────────────────────────────────
try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_recall_fscore_support, roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0f1117",
    "surface": "#1a1d27",
    "primary": "#6c63ff",
    "accent":  "#ff6584",
    "legit":   "#4fc3f7",
    "fraud":   "#ff6584",
    "text":    "#e0e0e0",
    "subtext": "#9e9e9e",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["subtext"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#2a2d3a",
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  PALETTE["surface"],
    "legend.edgecolor":  PALETTE["subtext"],
})


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """
    Compute classification metrics for binary fraud detection.

    Parameters
    ----------
    y_true  : Ground-truth labels (0 = legit, 1 = fraud)
    y_pred  : Predicted labels
    y_score : Predicted probability scores (for AUC)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, confusion_matrix
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available; skipping metrics.")
        return {}

    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = np.mean(y_true == y_pred)
    auc = (
        float(roc_auc_score(y_true, y_score))
        if y_score is not None and len(np.unique(y_true)) > 1
        else None
    )

    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info(f"\n{report}")

    metrics = {
        "accuracy":         float(acc),
        "precision":        float(prec),
        "recall":           float(rec),
        "f1":               float(f1),
        "auc":              auc,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def save_metrics(metrics: dict, output_dir: str = "."):
    """Save metrics dict to report_metrics.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")


# =============================================================================
# Risk Score Distribution
# =============================================================================

def plot_risk_distribution(
    dars_scores: np.ndarray,
    labels: np.ndarray,
    output_dir: str = ".",
):
    """
    Plot overlapping histograms of DARS scores for fraud vs. legitimate transactions.

    Parameters
    ----------
    dars_scores : np.ndarray of DARS values [0–100]
    labels      : np.ndarray of {0, 1}
    output_dir  : Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, 100, 50)
    legit_scores = dars_scores[labels == 0]
    fraud_scores = dars_scores[labels == 1]

    ax.hist(
        legit_scores, bins=bins, alpha=0.75,
        color=PALETTE["legit"], label=f"Legitimate (n={len(legit_scores):,})",
        edgecolor="none",
    )
    ax.hist(
        fraud_scores, bins=bins, alpha=0.85,
        color=PALETTE["fraud"], label=f"Fraud (n={len(fraud_scores):,})",
        edgecolor="none",
    )

    # Threshold line
    threshold = 60
    ax.axvline(threshold, color="#ffd700", linewidth=1.5, linestyle="--",
               label=f"Threshold ({threshold})")

    ax.set_xlabel("DARS Score", fontsize=12)
    ax.set_ylabel("Transaction Count", fontsize=12)
    ax.set_title("Digital Arrest Risk Score Distribution", fontsize=14, fontweight="bold",
                 color=PALETTE["text"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "risk_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Risk distribution plot saved to {path}")


# =============================================================================
# SHAP Feature Importance
# =============================================================================

def plot_feature_importance(
    xgb_model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    output_dir: str = ".",
    max_display: int = 15,
):
    """
    Generate a SHAP bar chart for the XGBoost shell account model.

    Parameters
    ----------
    xgb_model     : ShellAccountModel (or raw XGBClassifier)
    X             : Feature matrix used for SHAP explanation
    feature_names : Column names for X
    output_dir    : Output directory
    max_display   : Max features to display
    """
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed; skipping feature importance plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    clf = getattr(xgb_model, "clf", xgb_model)
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    n_show = min(max_display, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-n_show:][::-1]

    names = (
        [feature_names[i] for i in top_idx]
        if feature_names else [f"Feature {i}" for i in top_idx]
    )
    vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.45)))
    bars = ax.barh(
        names[::-1], vals[::-1],
        color=PALETTE["primary"], edgecolor="none", height=0.65,
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("SHAP Feature Importance – Shell Account Classifier",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, vals[::-1]):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8,
            color=PALETTE["subtext"],
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved to {path}")


# =============================================================================
# Transaction Network
# =============================================================================

def plot_transaction_network(
    edges: list[tuple[str, str]],
    node_scores: dict,
    output_dir: str = ".",
    threshold: float = 50.0,
    max_nodes: int = 200,
):
    """
    Visualize a sub-graph of transactions, coloring nodes by their DARS/SNS score.

    Parameters
    ----------
    edges       : List of (src, dst) account ID tuples
    node_scores : dict { account_id -> score [0–100] }
    output_dir  : Output directory
    threshold   : Score above which a node is flagged as high-risk
    max_nodes   : Max nodes to display (sample if graph is larger)
    """
    if not NX_AVAILABLE:
        logger.warning("networkx not installed; skipping network plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Limit size for visualization
    if len(G.nodes) > max_nodes:
        sampled_nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(sampled_nodes).copy()

    scores = [node_scores.get(n, 0.0) for n in G.nodes()]
    colors = [PALETTE["fraud"] if s > threshold else PALETTE["legit"] for s in scores]
    node_sizes = [max(30, min(300, s * 3)) for s in scores]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, len(G.nodes) ** 0.5))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=PALETTE["subtext"], alpha=0.4, arrows=True,
        arrowsize=8, width=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors, node_size=node_sizes, alpha=0.9,
    )

    # Legend proxies
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["fraud"], label=f"High Risk (score > {threshold:.0f})"),
        Patch(facecolor=PALETTE["legit"], label="Normal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_title("Transaction Graph – Account Risk Visualization",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "transaction_network.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Network plot saved to {path}")


# =============================================================================
# Fraud vs. Legitimate Boxplot
# =============================================================================

def plot_amount_comparison(
    df: pd.DataFrame,
    amount_col: str = "amount_inr",
    label_col:  str = "fraud_flag",
    output_dir: str = ".",
):
    """
    Side-by-side boxplot of transaction amounts for fraud vs. legitimate.

    Parameters
    ----------
    df          : UPI DataFrame with amount and label columns
    amount_col  : Name of the amount column
    label_col   : Binary label column (0/1)
    output_dir  : Output directory
    """
    if amount_col not in df.columns or label_col not in df.columns:
        logger.warning("amount_comparison skipped: required columns missing.")
        return

    os.makedirs(output_dir, exist_ok=True)

    legit_amounts = df.loc[df[label_col] == 0, amount_col].dropna()
    fraud_amounts = df.loc[df[label_col] == 1, amount_col].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        [legit_amounts, fraud_amounts],
        labels=["Legitimate", "Fraud"],
        patch_artist=True,
        notch=True,
        widths=0.4,
        medianprops=dict(color="#ffd700", linewidth=2),
        whiskerprops=dict(color=PALETTE["subtext"]),
        capprops=dict(color=PALETTE["subtext"]),
        flierprops=dict(marker="o", markersize=3, alpha=0.3, markeredgewidth=0),
    )
    bp["boxes"][0].set_facecolor(PALETTE["legit"])
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(PALETTE["fraud"])
    bp["boxes"][1].set_alpha(0.75)

    ax.set_ylabel("Transaction Amount (INR)", fontsize=11)
    ax.set_title("Transaction Amount: Fraud vs. Legitimate",
                 fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "amount_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Amount comparison plot saved to {path}")


# =============================================================================
# Score Component Dashboard
# =============================================================================

def plot_score_dashboard(
    df: pd.DataFrame,
    output_dir: str = ".",
):
    """
    Create a 2×2 dashboard showing all four DARS components.

    Parameters
    ----------
    df         : DataFrame with columns BRS, GRS, SNS, VA, DARS, fraud_flag
    output_dir : Output directory
    """
    required = ["BRS", "GRS", "SNS", "VA", "DARS", "fraud_flag"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"score_dashboard skipped: missing columns {missing}")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    components = [
        ("BRS", "Behavioral Risk Score",  PALETTE["primary"]),
        ("GRS", "Geographic Risk Score",  "#4caf50"),
        ("SNS", "Scam Network Score",     "#ff9800"),
        ("VA",  "Vulnerability Score",    "#e91e63"),
    ]

    for i, (col, title, color) in enumerate(components):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        bins = np.linspace(df[col].min(), df[col].max(), 40)
        ax.hist(df.loc[df["fraud_flag"] == 0, col], bins=bins,
                alpha=0.65, color=PALETTE["legit"], label="Legit", edgecolor="none")
        ax.hist(df.loc[df["fraud_flag"] == 1, col], bins=bins,
                alpha=0.85, color=color, label="Fraud", edgecolor="none")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("DARS Component Score Distributions",
                 fontsize=15, fontweight="bold", color=PALETTE["text"], y=1.01)

    plt.tight_layout()
    path = os.path.join(output_dir, "score_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Score dashboard saved to {path}")
