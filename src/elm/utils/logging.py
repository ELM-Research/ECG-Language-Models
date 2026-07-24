import sys
import traceback
import time
import wandb
from pathlib import Path
from typing import Optional, Union
import yaml
import argparse

from elm.utils.parallelism import is_main, barrier, broadcast_value

def setup_wandb(args, project = "ecg-bench-new", name = None):
    print("Initializing Wandb")
    wandb.init(project=project, config=args, name = name,)

def cleanup_wandb():
    error = sys.exc_info()[1]
    if error and wandb.run is not None:
        wandb.run.summary["error"] = traceback.format_exc()
    wandb.finish(exit_code=1 if error else 0)

def log_wandb(metrics, prefix = None):
    if prefix: metrics = {f"{prefix}/{k}" : v for k, v in metrics.items()}
    wandb.log(metrics)

def timeit(fn, desc="", dev=False):
    if not dev: return fn()
    start = time.time()
    out = fn()
    print(f"{desc} Timing: {time.time() - start:.4f}s")
    return out

def setup_experiment_folder(
    base_run_dir: str | Path,
    args: Namespace,
) -> Path:
    base_run_dir = Path(base_run_dir)

    if is_main():
        base_run_dir.mkdir(parents=True, exist_ok=True)
        run_id = next_run_id(base_run_dir)
        run_dir = base_run_dir / run_id
        run_dir.mkdir()
        save_config(run_dir, args)
    else:
        run_id = None

    run_id = broadcast_value(run_id, src=0)
    return base_run_dir / run_id


def next_run_id(base_run_dir: Path) -> str:
    run_ids = (
        int(path.name)
        for path in base_run_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    return str(max(run_ids, default=-1) + 1)


def save_config(run_dir: Path, args: Namespace) -> None:
    config = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }

    with (run_dir / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)