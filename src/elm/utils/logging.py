import sys
import traceback
import time
import wandb

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