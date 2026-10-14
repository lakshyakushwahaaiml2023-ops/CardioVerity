import logging
import sys
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("device")


def get_device(allow_cpu: bool = False) -> torch.device:
    """Validate compute device and log hardware details.

    Parameters
    ----------
    allow_cpu : bool, default=False
        If True, allows fallback to CPU when CUDA is unavailable.
        Can also be triggered by passing '--allow-cpu' in command-line arguments.

    Returns
    -------
    torch.device
        Device object for PyTorch computations ('cuda' or 'cpu').

    Raises
    ------
    RuntimeError
        If CUDA is unavailable and CPU fallback is not explicitly permitted.
    """
    cpu_allowed = allow_cpu or ("--allow-cpu" in sys.argv)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gib = vram_bytes / (1024 ** 3)
        logger.info("CUDA acceleration active: %s (Total VRAM: %.2f GiB)", device_name, vram_gib)
        return torch.device("cuda")

    if cpu_allowed:
        logger.warning("CUDA is unavailable. Running on CPU via '--allow-cpu' override.")
        return torch.device("cpu")

    error_msg = (
        "CUDA is unavailable. Under project rules, heavy compute must run on GPU. "
        "Verify NVIDIA driver / CUDA installation, or explicitly pass '--allow-cpu' to override."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)
