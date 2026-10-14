"""Resumable NFv2 relational-v1 matrix launcher.

The launcher creates one immutable coordinate list and dispatches independent
runner processes. It never reads external datasets and does not select a
configuration from target labels.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

# Import constants without duplicating the protocol values.
from CODE.experiment_diagnostics import run_nfv2_domain_fingerprint_relational_v1 as protocol


RUNNER = PROJECT_ROOT / "CODE" / "experiment_diagnostics" / "run_nfv2_domain_fingerprint_relational_v1.py"
DEFAULT_ROOT = PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_domain_fingerprint_relational_v1"


def matrix_coordinates(kind: str = "both") -> list[dict]:
    rows: list[dict] = []
    if kind in ("both", "domain"):
        for outer in protocol.DOMAINS:
            for representation in protocol.REPRESENTATIONS:
                for task in protocol.DOMAIN_TASKS:
                    for model in protocol.DOMAIN_MODELS:
                        for seed in protocol.SEEDS:
                            rows.append({
                                "kind": "domain", "outer": outer, "representation": representation,
                                "task": task, "model": model, "seed": seed,
                                "device": "cuda" if model == "mlp" else "cpu",
                            })
    if kind in ("both", "model"):
        for outer in protocol.DOMAINS:
            for representation in protocol.REPRESENTATIONS:
                for seed in protocol.SEEDS:
                    rows.append({
                        "kind": "model", "outer": outer, "representation": representation,
                        "seed": seed, "device": "cuda",
                    })
    return rows


def _queue_fingerprint(coords: list[dict], max_epochs: int, output_root: Path) -> str:
    payload = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "code_sha256": protocol.code_fingerprint(),
        "config_sha256": protocol.config_fingerprint(),
        "max_epochs": max_epochs,
        "output_root": str(output_root.resolve()),
        "coordinates": coords,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _command(row: dict, output_root: Path, max_epochs: int) -> list[str]:
    cmd = [sys.executable, str(RUNNER), "--kind", row["kind"], "--outer", row["outer"],
           "--representation", row["representation"], "--seed", str(row["seed"]),
           "--device", row["device"], "--output-root", str(output_root)]
    if row["kind"] == "domain":
        cmd.extend(["--task", row["task"], "--model", row["model"]])
    else:
        cmd.extend(["--max-epochs", str(max_epochs)])
    return cmd


def _run_one(row: dict, output_root: Path, log_dir: Path, max_epochs: int) -> dict:
    coord_id = "_".join(str(row[k]) for k in ("kind", "outer", "representation", "task", "model", "seed") if k in row)
    log_path = log_dir / f"{coord_id}.log"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(_command(row, output_root, max_epochs), cwd=PROJECT_ROOT,
                              env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {"coordinate": row, "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3), "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("both", "domain", "model"), default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--gpu-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=protocol.ERM_MAX_EPOCHS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_workers < 1 or args.gpu_workers < 1:
        raise SystemExit("worker counts must be positive")
    coords = matrix_coordinates(args.kind)
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "matrix_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "code_sha256": protocol.code_fingerprint(),
        "config_sha256": protocol.config_fingerprint(),
        "max_epochs": args.max_epochs,
        "coordinates": coords,
    }
    queue["queue_sha256"] = _queue_fingerprint(coords, args.max_epochs, args.output_root)
    (args.output_root / "matrix_queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"coordinates": len(coords), "queue_sha256": queue["queue_sha256"],
                      "cpu_workers": args.cpu_workers, "gpu_workers": args.gpu_workers}, indent=2))
    if args.dry_run:
        return 0

    cpu_rows = [row for row in coords if row["device"] == "cpu"]
    gpu_rows = [row for row in coords if row["device"] == "cuda"]
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
        futures = [pool.submit(_run_one, row, args.output_root, log_dir, args.max_epochs) for row in cpu_rows]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gpu_workers) as pool:
        futures = [pool.submit(_run_one, row, args.output_root, log_dir, args.max_epochs) for row in gpu_rows]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: json.dumps(item["coordinate"], sort_keys=True))
    (args.output_root / "matrix_run_summary.json").write_text(
        json.dumps({"queue_sha256": queue["queue_sha256"], "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    failures = [item for item in results if item["returncode"] != 0]
    print(json.dumps({"completed": len(results), "failed": len(failures)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
