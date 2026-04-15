import gc
import torch

from optimizers.optimizer_setup import get_optimizer

from dataloaders.build_dataloader import BuildDataLoader

from elms.build_elm import BuildELM

from runners.trainer import run_train
from runners.rl_trainer import run_rl_train

from utils.checkpoint_manager import CheckpointManager
from utils.seed_manager import set_seed
from utils.gpu_manager import is_main, init_dist, cleanup, GPUSetup, broadcast_value
from utils.dir_file_manager import setup_experiment_folders
from utils.wandb_manager import setup_wandb, cleanup_wandb

from configs.config import get_args
from configs.constants import RUNS_DIR

# This return true so I guess our Pytorch/Machine
# automatically detects for flash attention SDP
# print("flash attention SDP enabled.", torch.backends.cuda.flash_sdp_enabled())
torch.set_float32_matmul_precision("high")

def main():
    mode = "train"
    args = get_args(mode)
    args.mode = mode
    args.task = "train"

    if args.distributed:
        init_dist()

    gc.collect()
    torch.cuda.empty_cache()

    try:
        if not args.dev:
            data_name = "_".join(args.data)
            run_folder = setup_experiment_folders(
                f"{RUNS_DIR}/{args.llm}_{args.encoder}/{data_name}",
                args,
            )
        if is_main() and not args.dev:
            print(f"Run folder: {run_folder}")
            if args.wandb:
                setup_wandb(args)
        set_seed(args.seed)
        build_dataloader = BuildDataLoader(args)
        dataloader = build_dataloader.build_dataloader()
        args.max_steps = len(dataloader) * args.epochs
        build_elm = BuildELM(args)
        elm_components = build_elm.build_elm(dataloader.dataset.llm_tokenizer)
        gpu_setup = GPUSetup(args)
        elm = gpu_setup.setup_gpu(elm_components["elm"],
                                 elm_components["find_unused_parameters"])
        if args.dev:
            gpu_setup.print_model_device(elm, f"LLM: {args.llm} | ENCODER: {args.encoder} |")
        optimizer = get_optimizer(args, elm)
        if args.dev:
            checkpoint_manager = None
        else:
            checkpoint_manager = CheckpointManager(run_folder, args)
        start_epoch = 0
        if args.resume_ckpt and checkpoint_manager:
            start_epoch = checkpoint_manager.resume_checkpoint(args.resume_ckpt, elm, optimizer)
        runner = run_rl_train if getattr(args, "train_phase", "sft") == "rl" else run_train
        for epoch in range(start_epoch, args.epochs):
            train_result = runner(elm, optimizer, dataloader, epoch, args, checkpoint_manager)
            should_stop = False
            if checkpoint_manager and is_main():
                if checkpoint_manager.save_epoch(train_result["average_loss"]):
                    checkpoint_manager.save_checkpoint(elm, optimizer, epoch, -1, is_best=True, prefix="epoch_")
                if args.early_stopping and checkpoint_manager.stop_early():
                    print(f"Early stopping at epoch {epoch}")
                    should_stop = True
            should_stop = broadcast_value(should_stop, src=0)
            if should_stop:
                break

        if is_main() and not args.dev:
            with open(f"{run_folder}/DONE.txt", "w") as _:
                pass
    finally:
        if args.distributed:
            cleanup()
        if is_main() and args.wandb:
            cleanup_wandb()


if __name__ == "__main__":
    main()