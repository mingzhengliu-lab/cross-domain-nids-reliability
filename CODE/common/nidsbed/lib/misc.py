""":Miscellaneous helpers: seeding, class weights, file IO, time formatting."""
import os
import random
import json
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Seed Python / numpy / torch RNGs and force deterministic cuDNN."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(y_domain: np.ndarray, clip_max: float = 20.0,
                      device: str = 'cpu') -> torch.Tensor:
    """Per-domain inverse-frequency class weights for weighted BCE.

    The weight ratio ``w_attack / w_benign`` is the inverse class
    frequency, clipped to the symmetric range ``[1/clip_max, clip_max]``
    to prevent extreme up-weighting on heavily imbalanced domains.
    Weights are then normalized so that ``w_benign = 1``.

    Args:
        y_domain: (n,) integer labels (0 = benign, 1 = attack).
        clip_max: maximum allowed weight ratio. Default 20.
        device: torch device for the returned tensor.

    Returns:
        torch.Tensor of shape (2,): ``[w_benign, w_attack]``.
    """
    n0 = int((y_domain == 0).sum())
    n1 = int((y_domain == 1).sum())

    w0 = 1.0 / max(n0, 1)
    w1 = 1.0 / max(n1, 1)
    ratio = w1 / w0

    ratio = max(1.0 / clip_max, min(ratio, clip_max))

    w = torch.tensor([1.0, ratio], dtype=torch.float32)
    w = w * (2.0 / w.sum())  # normalize so mean class weight = 1

    return w.to(device)


def load_meta(data_dir: str) -> dict:
    """Load ``domains.json`` from a processed-data directory."""
    path = Path(data_dir) / "domains.json"
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj: dict, path: str, indent: int = 2):
    """Write a dict to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the trainable parameters of a torch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"
