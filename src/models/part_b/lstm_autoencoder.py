"""
VIDYUT Part B — LSTM Autoencoder for Anomaly Detection
===========================================================================
Stage 1 of the dual unsupervised theft detection pipeline.
Architecture: Encoder (LSTM 64 → 32) + Decoder (LSTM 32 → 64).
Flags consumers whose reconstruction error exceeds the 95th percentile
of the training distribution as anomalous.
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import get_settings
from src.utils.logger import get_logger

log = get_logger("vidyut.lstm_autoencoder")
settings = get_settings()

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("LSTM Autoencoder using device: %s", _DEVICE)


class _LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, n_features)  [reconstructed]
    """

    def __init__(
        self,
        n_features: int = 1,
        seq_len: int = 14,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=n_features,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        enc_out, (h_n, c_n) = self.encoder(x)
        # Take last hidden state and repeat for decoder
        latent = enc_out[:, -1:, :]              # (batch, 1, latent_dim)
        latent_rep = latent.repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        # Decode
        dec_out, _ = self.decoder(latent_rep)
        return dec_out  # (batch, seq_len, n_features)


class LSTMAutoencoderModel:
    """
    Wrapper around _LSTMAutoencoder with sklearn-style API.

    Anomaly threshold = 95th percentile of training reconstruction errors.
    """

    def __init__(
        self,
        seq_len: int = 14,
        n_features: int = 1,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 30,
        patience: int = 5,
    ) -> None:
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience

        self.net = _LSTMAutoencoder(n_features, seq_len, latent_dim, num_layers, dropout)
        self.net = self.net.to(_DEVICE)
        self.threshold_: Optional[float] = None
        self.train_losses_: List[float] = []
        self.is_fitted: bool = False

    def fit(self, sequences: np.ndarray) -> "LSTMAutoencoderModel":
        """
        Train the autoencoder on normal (unlabelled) consumption sequences.

        Parameters
        ----------
        sequences : np.ndarray, shape (N, seq_len, n_features)
        """
        log.info(
            "Training LSTM AE: %d sequences, seq_len=%d, latent=%d",
            len(sequences), self.seq_len, self.latent_dim,
        )
        X = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=2, factor=0.5
        )

        best_loss = float("inf")
        patience_counter = 0

        self.net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(_DEVICE)
                optimiser.zero_grad()
                recon = self.net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / len(sequences)
            self.train_losses_.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                log.info("Epoch %d/%d | Loss: %.6f", epoch + 1, self.n_epochs, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Compute anomaly threshold from training reconstruction errors
        recon_errors = self._compute_reconstruction_errors(sequences)
        percentile = settings.lstm_recon_percentile
        self.threshold_ = float(np.percentile(recon_errors, percentile))
        log.info(
            "LSTM AE threshold (p%d): %.6f", percentile, self.threshold_
        )
        self.is_fitted = True
        return self

    def _compute_reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Compute per-sequence mean squared reconstruction error."""
        self.net.eval()
        errors = []
        loader = DataLoader(
            TensorDataset(torch.tensor(sequences, dtype=torch.float32)),
            batch_size=512,
            shuffle=False,
        )
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(_DEVICE)
                recon = self.net(batch)
                mse = ((batch - recon) ** 2).mean(dim=[1, 2])
                errors.extend(mse.cpu().numpy().tolist())
        return np.array(errors)

    def predict_anomaly(
        self,
        sequences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly flags for a set of sequences.

        Returns
        -------
        anomaly_flags : np.ndarray of bool — True if anomalous
        recon_errors  : np.ndarray of float — reconstruction MSE per sequence
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        errors = self._compute_reconstruction_errors(sequences)
        flags = errors > self.threshold_
        return flags, errors

    def score_consumers(
        self,
        sequences: np.ndarray,
        consumer_ids: List[str],
    ) -> pd.DataFrame:
        """
        Aggregate sequence-level anomaly scores to consumer level.

        A consumer is flagged if ANY of their sequences is anomalous.

        Returns
        -------
        pd.DataFrame: consumer_id, max_recon_error, mean_recon_error, lstm_anomaly
        """
        flags, errors = self.predict_anomaly(sequences)
        df = pd.DataFrame({
            "consumer_id": consumer_ids,
            "recon_error": errors,
            "lstm_flag": flags,
        })
        agg = df.groupby("consumer_id").agg(
            max_recon_error=("recon_error", "max"),
            mean_recon_error=("recon_error", "mean"),
            lstm_anomaly=("lstm_flag", "any"),
        ).reset_index()
        return agg

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "threshold": self.threshold_,
            "config": {
                "seq_len": self.seq_len,
                "n_features": self.n_features,
                "latent_dim": self.latent_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }, path)
        log.info("LSTM AE saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "LSTMAutoencoderModel":
        ckpt = torch.load(path, map_location=_DEVICE)
        config = ckpt["config"]
        instance = cls(**config)
        instance.net.load_state_dict(ckpt["state_dict"])
        instance.net = instance.net.to(_DEVICE)
        instance.threshold_ = ckpt["threshold"]
        instance.is_fitted = True
        log.info("LSTM AE loaded from %s | threshold=%.6f", path, instance.threshold_)
        return instance
