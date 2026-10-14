import torch
import torch.nn as nn
import numpy as np


class PyTorchLogisticRegressionModule(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class GPULogisticRegression:
    """PyTorch Logistic Regression baseline trained on CUDA, exportable to CPU."""

    def __init__(
        self,
        in_features: int,
        lr: float = 0.02,
        weight_decay: float = 0.001,
        epochs: int = 200,
        device: str | torch.device = "cuda",
    ) -> None:
        self.in_features = in_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device(device)
        self.module = PyTorchLogisticRegressionModule(in_features).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "GPULogisticRegression":
        self.module.train()
        self.module.to(self.device)
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = self.module(x_tensor)
            loss = self.criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()

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

    def to_cpu(self) -> "GPULogisticRegression":
        self.device = torch.device("cpu")
        self.module.to(self.device)
        return self

    def save(self, file_path: str) -> None:
        torch.save({
            "in_features": self.in_features,
            "state_dict": self.module.state_dict(),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }, file_path)

    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "GPULogisticRegression":
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(
            in_features=checkpoint["in_features"],
            lr=checkpoint.get("lr", 0.02),
            weight_decay=checkpoint.get("weight_decay", 0.001),
            device=device,
        )
        model.module.load_state_dict(checkpoint["state_dict"])
        model.module.eval()
        return model
