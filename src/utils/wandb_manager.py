import wandb

def setup_wandb(args, name = None):
    print("Initializing Wandb")
    wandb.init(
        project="ecg-bench-new",
        config=args,
        name = name,
    )

def cleanup_wandb():
    wandb.finish()

def log_wandb(metrics, prefix = None):
    if prefix:
        metrics = {f"{prefix}/{k}" : v for k, v in metrics.items()}
    wandb.log(metrics)
