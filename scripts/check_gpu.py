import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.utils.device import get_device


def test_torch_cuda() -> bool:
    """Test PyTorch tensor matrix multiplication on CUDA."""
    try:
        import torch
        device = get_device()
        if device.type != "cuda":
            print("[FAIL] PyTorch CUDA Matmul: Resolved device is not CUDA")
            return False

        a = torch.randn(512, 512, device=device, dtype=torch.float32)
        b = torch.randn(512, 512, device=device, dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        assert c.is_cuda, "Result tensor is not on CUDA"
        assert not torch.isnan(c).any(), "Result tensor contains NaN"
        print("[PASS] PyTorch CUDA Matmul execution")
        return True
    except Exception as exc:
        print(f"[FAIL] PyTorch CUDA Matmul: {exc}")
        return False


def test_xgboost_cuda() -> bool:
    """Test XGBoost model fitting and prediction with device='cuda'."""
    try:
        import xgboost as xgb

        np.random.seed(42)
        x_dummy = np.random.randn(50, 6).astype(np.float32)
        y_dummy = np.random.randint(0, 2, size=50)

        clf = xgb.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            device="cuda",
            tree_method="hist",
            random_state=42
        )
        clf.fit(x_dummy, y_dummy)
        preds = clf.predict_proba(x_dummy)

        assert preds.shape == (50, 2), f"Unexpected predictions shape: {preds.shape}"
        assert not np.isnan(preds).any(), "Predictions contain NaN"
        print("[PASS] XGBoost CUDA Training (device='cuda')")
        return True
    except Exception as exc:
        print(f"[FAIL] XGBoost CUDA Training: {exc}")
        return False


def main() -> None:
    print("=" * 60)
    print("Verifying GPU Compute Environment")
    print("=" * 60)

    torch_passed = test_torch_cuda()
    xgb_passed = test_xgboost_cuda()

    print("-" * 60)
    if torch_passed and xgb_passed:
        print("RESULT: ALL GPU CHECKS PASSED")
        sys.exit(0)
    else:
        print("RESULT: GPU CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
