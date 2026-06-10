"""
main.py
-------
End-to-end CLI entry point for the DARS-GNN pipeline.

Usage
-----
    python main.py --upi_csv data/upi_transactions.csv \
                   --bank_xlsx data/bank_statements.xlsx \
                   [--output_dir results/] \
                   [--contamination 0.002] \
                   [--gat_epochs 100] \
                   [--threshold 60] \
                   [--device cpu] \
                   [--no_gat] \
                   [--no_xgb]

Pipeline Steps
--------------
1. Load UPI CSV and bank XLSX.
2. Feature engineering (behavioral, geographic, vulnerability, encoding).
3. Train IsolationForest → Behavioral Risk Score (BRS).
4. Build transaction graph → Train GAT → Scam Network Score (SNS).
5. Train XGBoost shell classifier → Shell Account Score (SAS).
6. Combine into Digital Arrest Risk Score (DARS).
7. Evaluate (metrics + visualizations).
8. Save all outputs.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# ── Configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DARS-GNN: Digital Arrest Risk Scoring Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--upi_csv",      required=True,  help="Path to UPI transactions CSV")
    parser.add_argument("--bank_xlsx",    required=True,  help="Path to bank statement XLSX")
    parser.add_argument("--output_dir",   default="results", help="Directory for output files")
    parser.add_argument("--contamination",type=float, default=0.002,
                        help="IsolationForest contamination parameter")
    parser.add_argument("--gat_epochs",   type=int,   default=80,
                        help="Number of GAT training epochs")
    parser.add_argument("--threshold",    type=float, default=60.0,
                        help="DARS score threshold for flagging fraud")
    parser.add_argument("--device",       default="cpu",
                        help="Device for PyTorch ('cpu' or 'cuda')")
    parser.add_argument("--no_gat",       action="store_true",
                        help="Skip GAT training (sets SNS = 0 for all)")
    parser.add_argument("--no_xgb",       action="store_true",
                        help="Skip XGBoost training (sets SAS = 0 for all)")
    parser.add_argument("--weights",      nargs=4, type=float,
                        metavar=("ALPHA", "BETA", "GAMMA", "DELTA"),
                        default=[0.40, 0.15, 0.35, 0.10],
                        help="DARS weights for BRS, GRS, SNS, VA (must sum to 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Add a log file handler
    fh = logging.FileHandler(os.path.join(args.output_dir, "pipeline.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 60)
    logger.info("  DARS-GNN Pipeline  –  Starting")
    logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load Data
    # ─────────────────────────────────────────────────────────────────────────
    from data_loader import load_upi_data, load_bank_data, construct_transaction_graph

    logger.info("[1/7] Loading data …")
    upi_df  = load_upi_data(args.upi_csv)
    bank_df = load_bank_data(args.bank_xlsx)

    logger.info(f"  UPI  -> {len(upi_df):,} transactions, {len(upi_df.columns)} columns")
    logger.info(f"  Bank -> {len(bank_df):,} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────
    from features import (
        compute_behavioral_features,
        compute_geographic_risk,
        compute_vulnerability,
        encode_categoricals,
        build_node_feature_matrix,
    )

    logger.info("[2/7] Feature engineering …")
    upi_df = compute_behavioral_features(upi_df)
    upi_df = compute_geographic_risk(upi_df)
    upi_df = compute_vulnerability(upi_df)

    cat_cols = [
        c for c in [
            "transaction_type", "merchant_category", "sender_bank",
            "receiver_bank", "network_type", "device_type", "sender_state",
        ]
        if c in upi_df.columns
    ]
    upi_df, encoders = encode_categoricals(upi_df, cat_cols)

    # Select numeric feature columns for IsolationForest
    brs_feature_cols = [
        c for c in [
            "amount_inr", "hour_of_day", "is_weekend",
            "tx_count", "avg_amt", "std_amt", "uniq_rec",
            "high_amount_flag", "hourly_tx_count", "GRS", "VA",
        ] + [f"{c}_enc" for c in cat_cols]
        if c in upi_df.columns
    ]

    X_brs = upi_df[brs_feature_cols].fillna(0).values
    logger.info(f"  BRS feature matrix: {X_brs.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Isolation Forest → BRS
    # ─────────────────────────────────────────────────────────────────────────
    from models import BehavioralAnomalyModel

    logger.info("[3/7] Training IsolationForest (BRS) …")
    iso_model = BehavioralAnomalyModel(contamination=args.contamination)
    iso_model.fit(X_brs)
    brs_scores = iso_model.score(X_brs)
    logger.info(f"  BRS  -> min={brs_scores.min():.2f}, max={brs_scores.max():.2f}, "
                f"mean={brs_scores.mean():.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Graph + GAT → SNS
    # ─────────────────────────────────────────────────────────────────────────
    from train import build_edge_index, map_sns_to_transactions

    logger.info("[4/7] Building transaction graph …")
    edges, node_feat_dict = construct_transaction_graph(bank_df)

    sns_per_txn = np.zeros(len(upi_df), dtype=np.float32)

    if not args.no_gat and len(edges) > 0:
        try:
            import torch
            from models import train_gat
            from features import build_node_feature_matrix

            node_X, node_ids, feat2idx = build_node_feature_matrix(node_feat_dict)
            edge_index = build_edge_index(edges, node_ids)

            if edge_index.shape[1] > 0:
                logger.info(f"[4/7] Training GAT on {len(node_ids)} nodes, "
                            f"{edge_index.shape[1]} edges...")
                gat_model, sns_scores = train_gat(
                    node_features=node_X,
                    edge_index=edge_index,
                    epochs=args.gat_epochs,
                    device=args.device,
                )
                # Map node-level SNS back to transaction rows
                sns_per_txn = map_sns_to_transactions(upi_df, node_ids, sns_scores)
                logger.info(f"  SNS  -> min={sns_per_txn.min():.2f}, "
                            f"max={sns_per_txn.max():.2f}")
            else:
                logger.warning("  No valid edges; SNS set to 0.")
                gat_model = None
        except Exception as e:
            logger.warning(f"  GAT training failed: {e}. SNS set to 0.")
            gat_model = None
    else:
        logger.info("  GAT skipped (--no_gat flag or no edges).")
        gat_model = None

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: XGBoost Shell Classifier → SAS
    # ─────────────────────────────────────────────────────────────────────────
    sas_scores = np.zeros(len(upi_df), dtype=np.float32)
    xgb_model  = None

    if not args.no_xgb and "fraud_flag" in upi_df.columns:
        y = upi_df["fraud_flag"].values
        if len(np.unique(y)) > 1:
            try:
                from models import ShellAccountModel
                from features import resample_smote

                logger.info("[5/7] Training XGBoost shell classifier (SAS) …")
                X_xgb = X_brs  # reuse same feature set

                # Augment with SNS if available
                if sns_per_txn.any():
                    X_xgb = np.column_stack([X_xgb, sns_per_txn])

                X_res, y_res = resample_smote(X_xgb, y)
                xgb_model = ShellAccountModel(
                    scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1))
                )
                xgb_model.fit(X_res, y_res)
                sas_scores = xgb_model.predict_proba(X_xgb) * 100.0
                logger.info(f"  SAS  -> min={sas_scores.min():.2f}, "
                            f"max={sas_scores.max():.2f}")
            except Exception as e:
                logger.warning(f"  XGBoost training failed: {e}. SAS set to 0.")
        else:
            logger.warning("  Only one class in fraud_flag; XGBoost skipped.")
    else:
        logger.info("  XGBoost skipped (--no_xgb flag or no labels).")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Combine → DARS
    # ─────────────────────────────────────────────────────────────────────────
    from train import combine_scores

    logger.info("[6/7] Computing DARS …")
    alpha, beta, gamma, delta = args.weights
    total = alpha + beta + gamma + delta
    if abs(total - 1.0) > 0.01:
        logger.warning(f"  Weights sum to {total:.3f} (not 1). Normalizing.")
        alpha, beta, gamma, delta = [w / total for w in [alpha, beta, gamma, delta]]

    weights = {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}

    grs_arr = upi_df["GRS"].fillna(0).values if "GRS" in upi_df.columns else np.zeros(len(upi_df))
    va_arr  = upi_df["VA"].fillna(0).values  if "VA"  in upi_df.columns else np.zeros(len(upi_df))

    dars_scores = combine_scores(brs_scores, grs_arr, sns_per_txn, va_arr, weights)
    logger.info(f"  DARS -> min={dars_scores.min():.2f}, max={dars_scores.max():.2f}, "
                f"mean={dars_scores.mean():.2f}")

    # Attach scores to DataFrame
    upi_df["BRS"]  = brs_scores
    upi_df["SNS"]  = sns_per_txn
    upi_df["SAS"]  = sas_scores
    upi_df["DARS"] = dars_scores
    upi_df["dars_flag"] = (upi_df["DARS"] >= args.threshold).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Evaluate + Visualize
    # ─────────────────────────────────────────────────────────────────────────
    from evaluate import (
        compute_metrics, save_metrics,
        plot_risk_distribution, plot_feature_importance,
        plot_transaction_network, plot_amount_comparison,
        plot_score_dashboard,
    )

    logger.info("[7/7] Evaluating and generating plots …")

    if "fraud_flag" in upi_df.columns and len(np.unique(upi_df["fraud_flag"])) > 1:
        y_true = upi_df["fraud_flag"].values
        y_pred = upi_df["dars_flag"].values
        metrics = compute_metrics(y_true, y_pred, y_score=dars_scores)
        save_metrics(metrics, output_dir=args.output_dir)
        logger.info(f"  Accuracy={metrics.get('accuracy', 0):.4f}  "
                    f"F1={metrics.get('f1', 0):.4f}  "
                    f"AUC={metrics.get('auc', 0):.4f}")

    # Visualizations
    plot_risk_distribution(dars_scores, upi_df.get("fraud_flag", pd.Series(0, index=upi_df.index)).values,
                           output_dir=args.output_dir)
    plot_amount_comparison(upi_df, output_dir=args.output_dir)
    plot_score_dashboard(upi_df, output_dir=args.output_dir)

    if xgb_model is not None:
        xgb_feat_names = brs_feature_cols + (["SNS"] if sns_per_txn.any() else [])
        try:
            plot_feature_importance(
                xgb_model, X_xgb,
                feature_names=xgb_feat_names,
                output_dir=args.output_dir,
            )
        except Exception as e:
            logger.warning(f"  SHAP plot failed: {e}")

    if edges and NX_OK():
        node_score_dict = {}
        for nid in set(s for s, _ in edges) | set(d for _, d in edges):
            if "sender_bank" in upi_df.columns:
                mask = upi_df["sender_bank"] == nid
                vals = upi_df.loc[mask, "DARS"].values
                node_score_dict[nid] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                node_score_dict[nid] = 0.0
        plot_transaction_network(edges, node_score_dict,
                                 output_dir=args.output_dir,
                                 threshold=args.threshold)

    # ─────────────────────────────────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────────────────────────────────
    from train import save_models

    output_csv = os.path.join(args.output_dir, "transactions_with_DARS.csv")
    upi_df.to_csv(output_csv, index=False)
    logger.info(f"  Scored transactions saved to {output_csv}")

    save_models(
        iso_model=iso_model,
        gat_model=gat_model,
        xgb_model=xgb_model,
        output_dir=args.output_dir,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    n_flagged = int(upi_df["dars_flag"].sum())
    pct = 100 * n_flagged / max(len(upi_df), 1)
    logger.info("=" * 60)
    logger.info("  DARS-GNN Pipeline - Complete")
    logger.info(f"  Transactions analysed : {len(upi_df):,}")
    logger.info(f"  Flagged (DARS >= {args.threshold:.0f}) : {n_flagged:,} ({pct:.2f}%)")
    logger.info(f"  Output directory       : {os.path.abspath(args.output_dir)}")
    logger.info("=" * 60)


def NX_OK():
    try:
        import networkx  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
