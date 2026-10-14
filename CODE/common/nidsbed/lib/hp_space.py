"""Hyperparameter search spaces for the LODO benchmark.

The API builds full method grids and samples deterministic per-method/per-target
subsets using stable SHA-256 based seeds rather than Python hash().
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Dict, List, Tuple

import numpy as np


# =========================================================================
# =========================================================================
GENERAL_GRID: Dict[str, list] = {
    'lr':           [3e-4, 1e-3, 3e-3],
    'weight_decay': [0, 1e-4, 1e-3],
    'dropout':      [0.2, 0.3, 0.4],
}

FIXED_HP: Dict[str, object] = {
    'batch_size': 4096,
    'hidden':     256,
    'feat_dim':   64,
}


# =========================================================================
# =========================================================================
METHOD_SPECIFIC_GRIDS: Dict[str, Dict[str, list]] = {
    'erm': {},
    'multitask_erm': {
        'family_lambda':      [0.1, 0.3, 1.0],
    },
    'irm': {
        'irm_lambda':        [0.1, 1.0, 10.0, 100.0],
        'irm_anneal_iters':  [0, 500, 1000],
    },
    'coral': {
        'coral_lambda':      [0.01, 0.1, 1.0, 10.0],
    },
    'deepcoral': {
        'coral_lambda':      [0.01, 0.1, 1.0, 10.0],
    },
    'groupdro': {
        'groupdro_eta':      [0.001, 0.01, 0.1],
    },
    'dann': {
        '_dann_lambda_setting': [
            ('fixed',    0.1),
            ('fixed',    1.0),
            ('schedule', None),
        ],
    },
    'fishr': {
        'fishr_lambda':        [0.01, 0.1, 1.0, 10.0, 100.0],
        'fishr_anneal_iters':  [0, 500, 1500],
    },
    'boda': {
        'boda_lambda':        [0.1, 1.0, 10.0],
        'boda_ema_momentum':  [0.9, 0.99],
    },
}


_DANN_DEFAULTS = {
    'disc_hidden': 64,
}

_BODA_DEFAULTS = {
    'boda_warmup_iters': 500,
    'boda_min_samples':  8,
}

# =========================================================================
# =========================================================================
def build_full_grid(method: str) -> List[dict]:
    """Return the full hyperparameter grid for ``method``.

    Combines the shared ``GENERAL_GRID`` (lr / weight_decay / dropout)
    with ``METHOD_SPECIFIC_GRIDS[method]`` and the ``FIXED_HP`` defaults.
    For DANN, the placeholder ``_dann_lambda_setting`` is expanded into
    ``dann_lambda_mode`` and ``dann_lambda_value``.
    """
    method = method.lower()
    if method not in METHOD_SPECIFIC_GRIDS:
        raise ValueError(f"unknown method: {method}. "
                         f"available: {sorted(METHOD_SPECIFIC_GRIDS)}")

    general_keys = list(GENERAL_GRID.keys())
    general_product = list(itertools.product(*[GENERAL_GRID[k] for k in general_keys]))

    method_grid = METHOD_SPECIFIC_GRIDS[method]
    if method_grid:
        method_keys = list(method_grid.keys())
        method_product = list(itertools.product(*[method_grid[k] for k in method_keys]))
    else:
        method_keys = []
        method_product = [()]

    full = []
    for gen_vals in general_product:
        for met_vals in method_product:
            hp = dict(FIXED_HP)
            for k, v in zip(general_keys, gen_vals):
                hp[k] = v
            for k, v in zip(method_keys, met_vals):
                hp[k] = v
            if '_dann_lambda_setting' in hp:
                mode, value = hp.pop('_dann_lambda_setting')
                hp['dann_lambda_mode'] = mode
                if mode == 'schedule':
                    hp['dann_lambda_value'] = 1.0
                else:
                    hp['dann_lambda_value'] = value
            if method == 'dann':
                for k, v in _DANN_DEFAULTS.items():
                    hp.setdefault(k, v)
            elif method == 'boda':
                for k, v in _BODA_DEFAULTS.items():
                    hp.setdefault(k, v)
            full.append(hp)

    return full


def hp_rng_seed(method: str, target: str) -> int:
    """Deterministic hash seed from (method, target)."""
    s = f"{method.lower()}|{target.lower()}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:8], 16)


def sample_hp_configs(
    method: str,
    target: str,
    n: int = 8,
) -> List[Tuple[int, dict]]:
    """Deterministic random sample from full grid."""
    full = build_full_grid(method)
    n_full = len(full)

    if n_full <= n:
        return list(enumerate(full))

    seed = hp_rng_seed(method, target)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_full, size=n, replace=False)
    indices = sorted(int(i) for i in indices)
    return [(hp_id, full[grid_idx]) for hp_id, grid_idx in enumerate(indices)]


def grid_info(method: str, target: str, n: int = 8) -> dict:
    """Build the hp_grid.json payload for one (method, target) cell.

    Reports the full grid size, the sampling mode (exhaustive when the
    grid fits in ``n`` configs, random otherwise) and the deterministic
    list of sampled HP configs with their ``hp_id``.
    """
    full = build_full_grid(method)
    n_full = len(full)
    is_exhaustive = n_full <= n

    configs = sample_hp_configs(method, target, n)
    seed = hp_rng_seed(method, target)

    return {
        'method':          method,
        'target':          target,
        'rng_seed_hex':    f"{seed:08x}",
        'full_grid_size':  n_full,
        'n_sampled':       len(configs),
        'sampling_mode':   'exhaustive' if is_exhaustive else 'random',
        'configs': [
            {'hp_id': hp_id, **hp}
            for hp_id, hp in configs
        ],
    }


# =========================================================================
# =========================================================================
if __name__ == '__main__':
    methods = ['erm', 'multitask_erm', 'irm', 'coral', 'groupdro', 'dann', 'fishr', 'boda']
    print(f"{'method':<12s} {'full_grid':>10s}  {'sampling':>10s}  example_hp")
    print("-" * 90)
    for m in methods:
        full = build_full_grid(m)
        configs = sample_hp_configs(m, 'unsw', n=8)
        mode = 'exhaustive' if len(full) <= 8 else 'random'
        example = configs[0][1]
        example_str = ", ".join(
            f"{k}={v}" for k, v in example.items()
            if k not in ('batch_size', 'hidden', 'feat_dim')
        )
        print(f"{m:<12s} {len(full):>10d}  {mode:>10s}  {example_str[:60]}")

    a = sample_hp_configs('fishr', 'unsw', n=8)
    b = sample_hp_configs('fishr', 'unsw', n=8)
    assert a == b, "sample_hp_configs must be deterministic for the same (method, target, n)"
    print("\n[OK] sample_hp_configs is deterministic for fishr/unsw")

    a_hps = [hp for _, hp in sample_hp_configs('fishr', 'unsw', 8)]
    b_hps = [hp for _, hp in sample_hp_configs('fishr', 'cic', 8)]
    common = sum(1 for hp in a_hps if hp in b_hps)
    print(
        f"[OK] (fishr,unsw) vs (fishr,cic) share {common} / 8 HP configs "
        f"(expected < 4)"
    )

    dann_full = build_full_grid('dann')
    modes = set((hp['dann_lambda_mode'], hp['dann_lambda_value']) for hp in dann_full)
    print(f"[OK] DANN lambda (mode, value) settings: {sorted(modes, key=str)}")
    assert len(modes) == 3, f"DANN lambda grid should expose 3 modes, got {len(modes)}"
