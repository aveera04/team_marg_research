"""
models.py
---------
Model components for the DARS-GNN pipeline:

  1. BehavioralAnomalyModel  - Isolation Forest for Behavioral Risk Score (BRS)
  2. GATNetwork              - Graph Attention Network for Scam Network Score (SNS)
  3. ShellAccountModel       - XGBoost classifier for Shell Account Score (SAS)
"""

import logging
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning(
        "torch not found. GATNetwork will be unavailable. "
        "Install with: pip install torch"
    )

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning(
        "torch_geometric not found. GATNetwork will be unavailable. "
        "Install with: pip install torch-geometric"
    )

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning(
        "xgboost not found. ShellAccountModel will be unavailable. "
        "Install with: pip install xgboost"
    )

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Behavioral Anomaly Model (Isolation Forest)
# =============================================================================

class BehavioralAnomalyModel:
    """
    Wrapper around scikit-learn IsolationForest to assign a
    Behavioral Risk Score (BRS) to each transaction.

    Higher BRS → more anomalous (potential fraud).
    Output is normalized to [0, 100].

    Parameters
    ----------
    n_estimators  : Number of trees in the forest (default 100).
    contamination : Expected fraction of outliers (default 0.002 ≈ 0.2%).
    random_state  : Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.002,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = MinMaxScaler(feature_range=(0, 100))
        self._fitted = False

    def fit(self, X: np.ndarray) -> "BehavioralAnomalyModel":
        """Fit the Isolation Forest on feature matrix X."""
        logger.info(f"Fitting IsolationForest on {X.shape[0]} samples...")
        self.model.fit(X)
        # Prefit scaler on training anomaly scores
        raw = -self.model.decision_function(X).reshape(-1, 1)
        self._scaler.fit(raw)
        self._fitted = True
        logger.info("IsolationForest training complete.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute BRS for each sample in X.

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in [0, 100].
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling score().")
        raw = -self.model.decision_function(X).reshape(-1, 1)
        scores = self._scaler.transform(raw).flatten()
        return np.clip(scores, 0, 100)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns -1 (anomaly) or 1 (normal) from IsolationForest.

        Returns
        -------
        np.ndarray of {-1, 1}.
        """
        return self.model.predict(X)


# =============================================================================
# 2. Graph Attention Network (GAT)
# =============================================================================

_GATBase = nn.Module if TORCH_AVAILABLE else object

class GATNetwork(_GATBase):
    """
    Two-layer Graph Attention Network (GAT) using PyTorch Geometric.

    Architecture:
      Layer 1: GATConv(in_channels → hidden_channels, heads=heads)  + ELU
      Layer 2: GATConv(hidden_channels*heads → hidden_channels, heads=1) + ELU
      Output: Node embedding vectors of size hidden_channels

    These embeddings are used to compute the Scam Network Score (SNS).

    Parameters
    ----------
    in_channels     : Number of input node features.
    hidden_channels : Hidden layer size (default 64).
    heads           : Number of attention heads (default 4).
    dropout         : Dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GATNetwork. "
                "Install: pip install torch-geometric"
            )
        super().__init__()
        self.dropout_p = dropout

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        """
        Forward pass.

        Parameters
        ----------
        x          : Node feature matrix  (n_nodes, in_channels)
        edge_index : Edge index tensor    (2, n_edges)

        Returns
        -------
        torch.Tensor of shape (n_nodes, hidden_channels) – node embeddings.
        """
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return x


def compute_sns_from_embeddings(embeddings) -> np.ndarray:
    """
    Derive Scam Network Score (SNS) from GAT node embeddings.

    Uses L2-norm of each node's embedding as an anomaly signal:
    nodes with atypical embeddings (far from the mean cluster) receive higher scores.

    Parameters
    ----------
    embeddings : torch.Tensor of shape (n_nodes, hidden_channels)

    Returns
    -------
    np.ndarray of shape (n_nodes,) with SNS values in [0, 100].
    """
    norms = torch.norm(embeddings, p=2, dim=1).cpu().detach().numpy()
    mn, mx = norms.min(), norms.max()
    sns = 100.0 * (norms - mn) / (mx - mn + 1e-9)
    return sns.astype(np.float32)


def train_gat(
    node_features: np.ndarray,
    edge_index,
    epochs: int = 100,
    lr: float = 0.005,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.1,
    device: str = "cpu",
) -> tuple:
    """
    Train the GAT in an unsupervised fashion using a reconstruction objective.

    Since ground-truth node labels may not be available, we use a
    self-supervised approach: the model is trained to reconstruct the
    feature matrix from embeddings (autoencoder-style).

    Parameters
    ----------
    node_features : np.ndarray  (n_nodes, n_features)
    edge_index    : torch.Tensor (2, n_edges)  – COO format
    epochs        : Number of training epochs (default 100)
    lr            : Learning rate (default 0.005)
    hidden_channels, heads, dropout : GATNetwork params
    device        : 'cpu' or 'cuda'

    Returns
    -------
    model      : Trained GATNetwork
    sns_scores : np.ndarray of SNS values for each node
    """
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric is required for train_gat().")

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Training GAT on device: {device}")

    x = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    in_channels = x.shape[1]
    model = GATNetwork(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    # Decoder: linear projection back to input dimension (reconstruction)
    decoder = nn.Linear(hidden_channels, in_channels).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        x_hat = decoder(emb)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"GAT Epoch {epoch:>3}/{epochs}  |  Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index)

    sns_scores = compute_sns_from_embeddings(embeddings)
    logger.info("GAT training complete.")
    return model, sns_scores


# =============================================================================
# 3. Shell Account Classifier (XGBoost)
# =============================================================================

class ShellAccountModel:
    """
    XGBoost binary classifier to identify shell/mule accounts.

    Output (SAS – Shell Account Score) is the predicted probability [0, 1]
    of a node being a shell account, scaled to [0, 100].

    Parameters
    ----------
    **params : Keyword arguments forwarded to XGBClassifier.
    """

    _DEFAULT_PARAMS = {
        "objective":             "binary:logistic",
        "n_estimators":          100,
        "max_depth":             5,
        "learning_rate":         0.1,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "random_state":          42,
        "eval_metric":           "auc",
        "n_jobs":                -1,
        "early_stopping_rounds": 10,   # XGBoost >= 2.0: goes in constructor
    }

    def __init__(self, **params):
        if not XGB_AVAILABLE:
            raise ImportError(
                "xgboost is required for ShellAccountModel. "
                "Install: pip install xgboost"
            )
        merged = {**self._DEFAULT_PARAMS, **params}
        self.clf = xgb.XGBClassifier(**merged)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list | None = None,
        early_stopping_rounds: int = 10,
    ) -> "ShellAccountModel":
        """
        Fit the XGBoost classifier.

        Parameters
        ----------
        X                     : Feature matrix (n_samples, n_features)
        y                     : Binary labels (n_samples,)
        eval_set              : Optional list of (X_val, y_val) for early stopping
        early_stopping_rounds : Rounds for early stopping (default 10)
        """
        if eval_set is None:
            eval_set = [(X, y)]

        logger.info(f"Fitting XGBoost on {X.shape[0]} samples...")
        self.clf.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self._fitted = True
        logger.info("XGBoost training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict shell account probability for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling predict_proba().")
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at a given probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_booster(self):
        """Return the underlying XGBoost booster (for SHAP, etc.)."""
        return self.clf
