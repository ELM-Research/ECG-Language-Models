import sys
import traceback
import time
import wandb
from pathlib import Path
import yaml

def setup_wandb(config, project = "ecg-bench-new", name = None):
    print("Initializing Wandb")
    wandb.init(project=project, config=config, name = name,)

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
    config: dict,
) -> Path:
    base_run_dir = Path(base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    run_ids = (
            int(path.name)
            for path in base_run_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    run_dir = base_run_dir / str(max(run_ids, default=-1) + 1)
    run_dir.mkdir()
    with (run_dir / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return run_dir