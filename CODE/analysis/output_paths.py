"""Keep generated presentation files separate from published inputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def output_directory(value):
    path = Path(value).expanduser().resolve()
    protected = [ROOT / name for name in ("CODE", "DATA", "RESULTS_PAPER", "paper", ".git")]
    if path == ROOT or any(path == p or path.is_relative_to(p) or p.is_relative_to(path)
                           for p in protected):
        raise ValueError("output must be separate from CODE, DATA, RESULTS_PAPER, paper, and .git")
    path.mkdir(parents=True, exist_ok=True)
    return path
