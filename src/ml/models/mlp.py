import copy
from typing import Any
import numpy as np
import torch
import torch.nn as nn


class PyTorchMLPModule(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.25) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GPUSmallMLP:
    """PyTorch MLP with dropout and early stopping trained on CUDA, exportable to CPU."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.25,
        lr: float = 0.002,
        weight_decay: float = 0.0005,
        max_epochs: int = 200,
        patience: int = 20,
        device: str | torch.device = "cuda",
    ) -> None:
        self.in_features = in_features
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = torch.device(device)
        self.module = PyTorchMLPModule(in_features, self.hidden_dims, dropout).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "GPUSmallMLP":
        self.module.train()
        self.module.to(self.device)

        x_tr = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            x_va = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_va = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.module.train()
            optimizer.zero_grad(set_to_none=True)
            logits = self.module(x_tr)
            loss = self.criterion(logits, y_tr)
            loss.backward()
            optimizer.step()

            if has_val:
                self.module.eval()
                with torch.no_grad():
                    val_logits = self.module(x_va)
                    val_loss = self.criterion(val_logits, y_va).item()

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.module.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.module.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.module.eval()
        dev = next(self.module.parameters()).device
        x_tensor = torch.tensor(X, dtype=torch.float32, device=dev)
        with torch.no_grad():
            logits = self.module(x_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy()
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def to_cpu(self) -> "GPUSmallMLP":
        self.device = torch.device("cpu")
        self.module.to(self.device)
        return self

    def save(self, file_path: str) -> None:
        torch.save({
            "in_features": self.in_features,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "state_dict": self.module.state_dict(),
        }, file_path)

    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "GPUSmallMLP":
        ckpt = torch.load(file_path, map_location=device)
        model = cls(
            in_features=ckpt["in_features"],
            hidden_dims=ckpt["hidden_dims"],
            dropout=ckpt["dropout"],
            device=device,
        )
        model.module.load_state_dict(ckpt["state_dict"])
        model.module.eval()
        return model
