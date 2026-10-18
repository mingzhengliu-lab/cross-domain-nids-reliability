from pathlib import Path
import hashlib
import json
import numpy as np


def load_indices(data_dir):
    data_dir = Path(data_dir)
    path = Path(__file__).resolve().parents[2]/'DATA/nfv2_processed/splits/sensitivity_indices.npz'
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(path) != manifest['file_sha256']:
        raise ValueError('Sensitivity index archive does not match its manifest')
    for name, digest in manifest['data_inputs'].items():
        if sha(data_dir/name) != digest:
            raise ValueError(f'Sensitivity indices require the frozen positional data keys: {name}')
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def check_groups(indices, train, test, group_ids):
    indices = np.asarray(indices, dtype=np.int64)
    if not np.isin(indices, train).all() or len(np.unique(group_ids[indices])) != len(indices):
        raise ValueError('Expected one representative per target training group')
    if np.intersect1d(group_ids[indices], group_ids[test]).size:
        raise ValueError('Selected target groups overlap the evaluation partition')


def group_weights(group_ids):
    _, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    return 1. / counts[inverse]
