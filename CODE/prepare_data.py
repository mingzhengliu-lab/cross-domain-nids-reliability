"""Rebuild the paper's float32 arrays using frozen positional row indices.

Input: the original four NFv2 Parquet files, in their archived row order.
The default output is separate from the original DATA. --verify compares to
existing arrays without writing them. CSV downloads may have a different row
order or numeric representation; use this check before applying frozen keys.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FILES = {'unsw':'NF-UNSW-NB15-V2.parquet', 'cic':'NF-CSE-CIC-IDS2018-V2.parquet',
         'bot':'NF-BoT-IoT-V2.parquet', 'ton':'NF-ToN-IoT-V2.parquet'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rebuild(domain, raw, keys, output, verify=False):
    path=raw/FILES[domain]
    raw_digest=sha256(path)
    reference=keys/'raw_data_manifest.json'
    if reference.exists():
        record=next(r for r in json.loads(reference.read_text(encoding='utf-8')) if r['domain']==domain)
        if raw_digest!=record['raw_sha256'] or sha256(keys/f'{domain}_row_id.npy')!=record['row_indices_sha256']:
            raise ValueError(f'{domain}: raw file or positional keys do not match the archived fingerprints; do not apply frozen splits to a reordered download')
    manifest=json.loads((keys/'domains.json').read_text(encoding='utf-8'))
    columns=manifest['feature_names']
    ids=np.load(keys/f'{domain}_row_id.npy')
    if len(np.unique(ids))!=len(ids): raise ValueError('duplicate positional row index')
    parquet=pq.ParquetFile(path)
    if ids.min()<0 or ids.max()>=parquet.metadata.num_rows: raise ValueError('row index exceeds raw dataset')
    order=np.argsort(ids); sorted_ids=ids[order]
    X=np.empty((len(ids),len(columns)),dtype=np.float32)
    y=np.empty(len(ids),dtype=np.int64)
    attack=np.empty(len(ids),dtype=object)
    offset=0
    for batch in parquet.iter_batches(batch_size=250000,columns=columns+['Label','Attack']):
        end=offset+batch.num_rows
        start_at,end_at=np.searchsorted(sorted_ids,[offset,end])
        if end_at>start_at:
            selected=sorted_ids[start_at:end_at]-offset
            dest=order[start_at:end_at]
            # Arrow-to-Pandas conversion follows the archived column numeric
            # interpretation; NaN/Inf remain present for source-only handling.
            frame=batch.to_pandas().iloc[selected]
            X[dest]=frame[columns].to_numpy(dtype=np.float32)
            y[dest]=frame['Label'].to_numpy(dtype=np.int64)
            attack[dest]=frame['Attack'].to_numpy(dtype=object)
        offset=end
    if verify:
        oldX=np.load(keys/f'{domain}_X.npy',mmap_mode='r')
        oldy=np.load(keys/f'{domain}_y.npy')
        olda=np.load(keys/f'{domain}_attack.npy',allow_pickle=True)
        if not np.array_equal(X.view(np.uint32),oldX.view(np.uint32)):
            diff=(X.view(np.uint32)!=oldX.view(np.uint32))
            raise ValueError(f'{domain}: {int(diff.sum())} feature bit patterns differ; frozen row order is not portable to this raw file')
        if not np.array_equal(y,oldy) or not np.array_equal(attack,olda):
            raise ValueError(f'{domain}: label or attack metadata differs')
        print(f'{domain}: raw file + frozen row indices exactly reproduce features and labels',flush=True)
    else:
        output.mkdir(parents=True,exist_ok=True)
        arrays=[('X',X),('y',y),('attack',attack)]
        if output.resolve()!=keys.resolve(): arrays.append(('row_id',ids))
        for suffix,arr in arrays:
            p=output/f'{domain}_{suffix}.npy'
            if p.exists(): raise FileExistsError(f'refusing to overwrite {p}')
            np.save(p,arr)
        print(f'{domain}: rebuilt {len(ids):,} rows',flush=True)
    return dict(domain=domain,raw_filename=path.name,raw_rows=parquet.metadata.num_rows,
        raw_sha256=raw_digest,row_indices_sha256=sha256(keys/f'{domain}_row_id.npy'),
        feature_dtype='float32',feature_count=len(columns),sample_rows=len(ids),
        exact_existing_arrays_verified=verify)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=ROOT/'DATA/nfv2_raw')
    p.add_argument('--keys-dir',type=Path,default=ROOT/'DATA/nfv2_processed')
    p.add_argument('--output',type=Path,default=ROOT/'RESULTS/reruns/rebuilt_data')
    p.add_argument('--domains',nargs='+',choices=FILES,default=list(FILES))
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if not args.verify:
        existing=[args.output/f'{d}_{s}.npy' for d in args.domains for s in ('X','y','attack') if (args.output/f'{d}_{s}.npy').exists()]
        if existing: raise FileExistsError(f'refusing to overwrite existing data arrays: {existing[0]}')
    rows=[rebuild(d,args.raw_dir,args.keys_dir,args.output,args.verify) for d in args.domains]
    report=ROOT/'RESULTS/reruns/submission_checks/raw_data' if args.verify else args.output
    report.mkdir(parents=True,exist_ok=True)
    (report/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    if not args.verify and args.output.resolve()!=args.keys_dir.resolve():
        import shutil
        for f in ('domains.json','feature_names.txt','raw_data_manifest.json'):
            if not (args.keys_dir/f).exists(): continue
            shutil.copy2(args.keys_dir/f,args.output/f)
        for domain in args.domains:
            p=args.keys_dir/f'{domain}_duplicate_group_id.npy'
            if p.exists(): shutil.copy2(p,args.output/p.name)
        shutil.copytree(args.keys_dir/'splits',args.output/'splits',dirs_exist_ok=True)

if __name__=='__main__': main()
