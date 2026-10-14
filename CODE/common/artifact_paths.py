"""Resolve original result locations inside the distributed paper project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPINGS = (
    ("/root/rivermind-data/paper/results/stage3/", "RESULTS/stage3_lodo/stage3/"),
    ("/root/rivermind-data/paper/results/revision_fewshot_v5/", "RESULTS/low_label/revision_fewshot_v5/"),
    ("/root/rivermind-fs/nfv2_target_label_budget_confirmation_v1/", "RESULTS/low_label/target_label_budget_confirmation/"),
    ("/root/rivermind-data/paper/results/tabular_baselines/", "RESULTS/model_family/tabular_baselines/"),
    ("/root/rivermind-data/paper/results/anomaly/", "RESULTS/model_family/anomaly/"),
)

def resolve_artifact(value, root=PROJECT_ROOT):
    text = str(value).replace("\\", "/")
    for old, new in MAPPINGS:
        if text.startswith(old):
            return Path(root) / new / text[len(old):]
    for prefix in ("/root/rivermind-data/target_reliability_paper/",
                   "/root/autodl-tmp/target_reliability_paper/"):
        if text.startswith(prefix):
            return Path(root) / text[len(prefix):]
    path = Path(text)
    return path if path.is_absolute() else Path(root) / path
