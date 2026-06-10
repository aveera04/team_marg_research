# DARS-GNN: Digital Arrest Risk Scoring Pipeline

A complete end-to-end fraud detection pipeline for UPI transactions using Graph Neural Networks, Isolation Forest, and XGBoost.

---

## 📁 Project Structure

```
team_marg_research/
├── data_loader.py      # Load & preprocess UPI CSV and Bank XLSX
├── features.py         # Feature engineering (behavioral, geo, vulnerability, SMOTE)
├── models.py           # IsolationForest, GATNetwork, XGBoostClassifier
├── train.py            # DARS score combination, edge-index building, model saving
├── evaluate.py         # Metrics, plots (risk distribution, SHAP, network graph)
├── main.py             # CLI entry point – runs the full pipeline
├── utils.py            # Utility helpers + unit tests
├── requirements.txt    # Python dependencies
└── results/            # Auto-created: scores, plots, saved models
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> **GPU note**: For GAT training on large graphs, install the CUDA build of PyTorch and torch-geometric.

---

## 🚀 Running the Pipeline

```bash
python main.py \
  --upi_csv   data/upi_transactions.csv \
  --bank_xlsx data/bank_statements.xlsx \
  --output_dir results/
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--upi_csv` | *(required)* | Path to UPI transactions CSV |
| `--bank_xlsx` | *(required)* | Path to bank statement XLSX |
| `--output_dir` | `results/` | Directory for all outputs |
| `--contamination` | `0.002` | IsolationForest outlier fraction |
| `--gat_epochs` | `80` | GAT training epochs |
| `--threshold` | `60` | DARS score to flag as fraud |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--no_gat` | — | Skip GAT (fast mode) |
| `--no_xgb` | — | Skip XGBoost |
| `--weights α β γ δ` | `0.40 0.15 0.35 0.10` | DARS component weights |

---

## 🧠 DARS Formula

```
DARS = α·BRS + β·GRS + γ·SNS + δ·VA
```

| Component | Model | Weight |
|-----------|-------|--------|
| **BRS** – Behavioral Risk Score | Isolation Forest | α = 0.40 |
| **GRS** – Geographic Risk Score | Heuristic (state mismatch) | β = 0.15 |
| **SNS** – Scam Network Score | Graph Attention Network | γ = 0.35 |
| **VA**  – Vulnerability Assessment | Age + device heuristic | δ = 0.10 |

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `transactions_with_DARS.csv` | All transactions with BRS, GRS, SNS, VA, DARS, dars_flag |
| `risk_distribution.png` | DARS score histogram (fraud vs legit) |
| `score_dashboard.png` | 2×2 component score distributions |
| `feature_importance.png` | SHAP bar chart (XGBoost) |
| `transaction_network.png` | Account graph (red = high risk) |
| `amount_comparison.png` | Amount boxplot (fraud vs legit) |
| `report_metrics.json` | Precision, Recall, F1, AUC |
| `iso_forest_model.pkl` | Saved IsolationForest |
| `gat_model.pth` | Saved GAT weights |
| `xgb_shell_model.json` | Saved XGBoost model |
| `pipeline.log` | Full run log |

---

## 🧪 Running Tests

```bash
python utils.py
```

---

## 📋 Expected Input Schema

### UPI CSV

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique identifier |
| `timestamp` | datetime | Transaction time |
| `transaction_type` | string | P2P, P2M, etc. |
| `amount (INR)` | float | Amount in INR |
| `sender_age_group` | string | e.g. `26-35` |
| `sender_state` | string | e.g. `Maharashtra` |
| `sender_bank` | string | e.g. `HDFC Bank` |
| `receiver_bank` | string | Receiver's bank |
| `device_type` | string | Android / iOS |
| `fraud_flag` | int (0/1) | Ground truth label |

### Bank XLSX

| Column | Type | Description |
|--------|------|-------------|
| `Account No` | string | Account number |
| `DATE` | datetime | Transaction date |
| `TRANSACTION DETAILS` | string | e.g. "TRF FROM XYZ" |
| `WITHDRAWAL AMT` | float | Debit amount |
| `DEPOSIT AMT` | float | Credit amount |
| `BALANCE AMT` | float | Running balance |

---

## ⚡ Quick Test (no real data)

```bash
# Skip GAT and XGBoost, generate synthetic-friendly run
python main.py --upi_csv data/sample_upi.csv --bank_xlsx data/sample_bank.xlsx --no_gat --no_xgb
```
