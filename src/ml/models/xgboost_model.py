from typing import Any
import numpy as np
import xgboost as xgb


class GPUXGBoostModel:
    """XGBoost binary classifier configured for CUDA training and CPU serve-time inference."""

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        device: str = "cuda",
        random_state: int = 42,
    ) -> None:
        self.device = device
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "device": device,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        self.model = xgb.XGBClassifier(**self.params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        verbose: bool = False,
    ) -> "GPUXGBoostModel":
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def to_cpu(self) -> "GPUXGBoostModel":
        """Reconfigure underlying model for CPU inference at serve time."""
        self.model.set_params(device="cpu")
        return self

    def save(self, file_path: str) -> None:
        self.model.save_model(file_path)

    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "GPUXGBoostModel":
        wrapper = cls(device=device)
        wrapper.model.load_model(file_path)
        wrapper.model.set_params(device=device)
        return wrapper
