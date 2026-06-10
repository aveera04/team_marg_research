"""
data_loader.py
--------------
Handles loading, cleaning, and preprocessing of:
  - UPI transaction CSV
  - Bank statement XLSX (multi-account format)
Also constructs a directed transaction graph from bank data.
"""

import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UPI Data Loader
# ---------------------------------------------------------------------------

def load_upi_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the UPI transactions CSV.

    Steps:
      - Read CSV with date parsing
      - Standardize column names
      - Drop rows missing essential fields
      - Derive temporal features (hour_of_day, day_of_week, is_weekend)

    Parameters
    ----------
    csv_path : str
        Path to the UPI transactions CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned UPI transaction DataFrame.
    """
    logger.info(f"Loading UPI data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names: strip, lowercase, replace spaces with underscores,
    # remove parentheses
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()\/]", "", regex=True)
    )

    logger.info(f"Columns after normalization: {list(df.columns)}")

    # Parse timestamp column if present
    ts_col = _find_column(df, ["timestamp", "date", "transaction_date", "txn_date"])
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        # Derive temporal features
        if "hour_of_day" not in df.columns:
            df["hour_of_day"] = df[ts_col].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df[ts_col].dt.day_name()
        if "is_weekend" not in df.columns:
            df["is_weekend"] = df[ts_col].dt.dayofweek.isin([5, 6]).astype(int)

    # Find and normalize amount column
    amt_col = _find_column(df, ["amount_inr", "amount", "txn_amount", "transaction_amount"])
    if amt_col and amt_col != "amount_inr":
        df.rename(columns={amt_col: "amount_inr"}, inplace=True)

    # Find transaction id column
    id_col = _find_column(df, ["transaction_id", "transaction_id", "txn_id", "id"])
    if id_col and id_col != "transaction_id":
        df.rename(columns={id_col: "transaction_id"}, inplace=True)

    # Drop rows missing essential fields
    essential = [c for c in ["transaction_id", "amount_inr"] if c in df.columns]
    if essential:
        before = len(df)
        df = df.dropna(subset=essential)
        logger.info(f"Dropped {before - len(df)} rows with missing essential fields.")

    # Ensure amount is numeric
    if "amount_inr" in df.columns:
        df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce").fillna(0.0)

    # Ensure fraud_flag exists
    if "fraud_flag" not in df.columns:
        logger.warning("'fraud_flag' column not found. Defaulting to 0.")
        df["fraud_flag"] = 0
    else:
        df["fraud_flag"] = pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0).astype(int)

    logger.info(f"UPI data loaded: {len(df)} rows, {len(df.columns)} columns.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bank Statement Loader
# ---------------------------------------------------------------------------

def load_bank_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load and preprocess the bank statement XLSX (multi-account format).

    The Excel file may have multiple accounts concatenated vertically.
    Each account block starts with a row containing the Account No.

    Steps:
      - Read all rows as strings to avoid type coercion
      - Normalize column names
      - Forward-fill Account No across account blocks
      - Parse dates and numeric amounts
      - Remove helper/empty rows

    Parameters
    ----------
    xlsx_path : str
        Path to the bank statement XLSX file.

    Returns
    -------
    pd.DataFrame
        Cleaned bank transaction DataFrame.
    """
    logger.info(f"Loading bank data from: {xlsx_path}")
    df = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # Drop unnamed trailing columns (e.g. the stray '.' column)
    drop_cols = [c for c in df.columns if re.fullmatch(r"\.+|Unnamed.*", c)]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info(f"Dropped trailing columns: {drop_cols}")

    # Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Forward-fill Account No so each transaction row has its account
    acc_col = _find_column(df, ["account_no", "account no", "account_number", "acc_no"])
    if acc_col:
        df[acc_col] = df[acc_col].ffill()
        # Strip stray quotes from account numbers
        df[acc_col] = df[acc_col].str.strip("'\" ")
        df.rename(columns={acc_col: "account_no"}, inplace=True)
    else:
        logger.warning("Account No column not found in bank data.")
        df["account_no"] = "UNKNOWN"

    # Normalize remaining column names
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Parse date column
    date_col = _find_column(df, ["DATE", "TXN_DATE", "TRANSACTION_DATE", "VALUE_DATE"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.dropna(subset=[date_col], inplace=True)
        if date_col != "DATE":
            df.rename(columns={date_col: "DATE"}, inplace=True)

    # Parse numeric amount columns
    for col in ["WITHDRAWAL_AMT", "DEPOSIT_AMT", "BALANCE_AMT",
                "WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]:
        clean = col.replace(" ", "_")
        if col in df.columns:
            df.rename(columns={col: clean}, inplace=True)
            df[clean] = pd.to_numeric(df[clean], errors="coerce").fillna(0.0)

    logger.info(f"Bank data loaded: {len(df)} rows.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def construct_transaction_graph(bank_df: pd.DataFrame):
    """
    Parse bank transaction descriptions to create a directed edge list
    representing fund transfers between accounts.

    Heuristics used:
      - 'TRF FROM <X>'  -> edge (X -> this_account)
      - 'TRF TO <X>'    -> edge (this_account -> X)
      - 'IMPS/NEFT/UPI' with a recognizable pattern

    Parameters
    ----------
    bank_df : pd.DataFrame
        Cleaned bank statement DataFrame from load_bank_data().

    Returns
    -------
    edges : list of (str, str)
        Directed edges as (source_account, destination_account).
    node_features : dict
        Mapping from account_id -> dict of aggregated features.
    """
    edges = []
    node_features: dict = {}

    detail_col = _find_column(
        bank_df,
        ["TRANSACTION_DETAILS", "TRANSACTION DETAILS", "NARRATION", "DESCRIPTION", "DETAILS"]
    )

    for _, row in bank_df.iterrows():
        acc = str(row.get("ACCOUNT_NO", row.get("account_no", "UNK"))).strip("'\" ")
        details = str(row.get(detail_col, "") if detail_col else "").upper()

        src, dst = None, None

        if "TRF FROM" in details or "TRANSFER FROM" in details:
            tokens = details.split()
            idx = next(
                (i for i, t in enumerate(tokens) if t in ("FROM",)), len(tokens) - 1
            )
            counterparty = tokens[idx + 1] if idx + 1 < len(tokens) else "UNKNOWN"
            src, dst = counterparty, acc

        elif "TRF TO" in details or "TRANSFER TO" in details:
            tokens = details.split()
            idx = next(
                (i for i, t in enumerate(tokens) if t in ("TO",)), len(tokens) - 1
            )
            counterparty = tokens[idx + 1] if idx + 1 < len(tokens) else "UNKNOWN"
            src, dst = acc, counterparty

        elif any(kw in details for kw in ["IMPS", "NEFT", "RTGS", "UPI"]):
            # Generic: treat as intra-account or unknown counterparty
            src, dst = acc, "EXTERNAL"

        if src and dst and src != dst:
            edges.append((src, dst))

        # Accumulate node features for the account
        dep = float(row.get("DEPOSIT_AMT", 0) or 0)
        wit = float(row.get("WITHDRAWAL_AMT", 0) or 0)
        bal = float(row.get("BALANCE_AMT", 0) or 0)

        nf = node_features.setdefault(acc, {
            "total_deposit": 0.0,
            "total_withdrawal": 0.0,
            "tx_count": 0,
            "last_balance": 0.0,
        })
        nf["total_deposit"] += dep
        nf["total_withdrawal"] += wit
        nf["tx_count"] += 1
        nf["last_balance"] = bal if bal != 0 else nf["last_balance"]

    # Compute net_flow_ratio: |deposit - withdrawal| / (deposit + withdrawal + 1e-9)
    for acc, feats in node_features.items():
        total = feats["total_deposit"] + feats["total_withdrawal"] + 1e-9
        feats["net_flow_ratio"] = abs(feats["total_deposit"] - feats["total_withdrawal"]) / total

    logger.info(
        f"Graph constructed: {len(edges)} edges, {len(node_features)} account nodes."
    )
    return edges, node_features


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in col_map:
            return col_map[key]
    return None
