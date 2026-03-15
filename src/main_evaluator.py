import gc
import json
import os
import torch
from pathlib import Path

from configs.config import get_args
from utils.gpu_manager import GPUSetup, init_dist, cleanup, is_main
from utils.seed_manager import set_seed
from dataloaders.build_dataloader import BuildDataLoader
from elms.build_elm import BuildELM
from runners.evaluator import evaluate, run_statistical_analysis, save_confusion_matrix_png, \
                                save_other_outputs_histogram_png, save_incorrect_predictions_histogram_png

def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "eval"
    args = get_args(mode)
    args.mode = mode

    if getattr(args, "distributed", False):
        init_dist()

    # folds = ["1", "2", "3", "4", "5"]
    # seeds = [1337, 1338, 1339, 1340, 1341]
    folds = ["1"]
    seeds = [1337, 1338]
    all_metrics = []

    if args.elm_ckpt:
        checkpoint_dir = os.path.dirname(args.elm_ckpt)
        ckpt_file_name = Path(args.elm_ckpt).stem
    else:
        checkpoint_dir = "./"
        ckpt_file_name = "no_ckpt"
    sys_prompt_name = Path(args.system_prompt).stem
    data_name = "_".join(args.data)
    results_file = os.path.join(checkpoint_dir, f"{ckpt_file_name}_{data_name}_{sys_prompt_name}_{args.perturb}.json")
    debug_path = results_file.replace(".json", "_debug.txt")
    debug_file = open(debug_path, "w") if is_main() else None
    for fold in folds:
        for seed in seeds:
            if is_main():
                print(f"Evaluating fold {fold} with seed {seed}")
            args.fold = fold
            args.seed = seed
            set_seed(args.seed)
            build_dataloader = BuildDataLoader(args)
            dataloader = build_dataloader.build_dataloader()
            build_elm = BuildELM(args)
            elm_components = build_elm.build_elm(dataloader.dataset.llm_tokenizer)
            gpu_setup = GPUSetup(args)
            elm = gpu_setup.setup_gpu(elm_components["elm"], elm_components["find_unused_parameters"])
            if args.dev:
                gpu_setup.print_model_device(elm, f"{args.llm}_{args.encoder}")
            out = evaluate(elm, dataloader, args, debug_file=debug_file)
            all_metrics.append(out)
            if is_main() and len(all_metrics) == 1:
                examples_path = results_file.replace(".json", "_examples.json")
                examples = [{"prompt": p, "predicted": h, "ground_truth": r}
                            for p, h, r in zip(out["prompts"], out["hypotheses"], out["references"])]
                with open(examples_path, "w") as ef:
                    json.dump(examples, ef, indent=2)
                print(f"Saved {len(examples)} eval examples to {examples_path}")
        if is_main():
            if "confusion_matrix" in out:
                cm_path = results_file.replace(".json", f"{fold}_{seed}.png")
                save_confusion_matrix_png(out["confusion_matrix"], cm_path)
                other_path = results_file.replace(".json", f"{fold}_{seed}_other.png")
                save_other_outputs_histogram_png(out["other_output_counts"], other_path, top_k = 10)
            incorrect_path = results_file.replace(".json", f"{fold}_{seed}_incorrect.png")
            save_incorrect_predictions_histogram_png(out["references"], out["hypotheses"], incorrect_path)
    if debug_file is not None:
        debug_file.close()
        if is_main():
            print(f"Saved debug dump to {debug_path}")
    if is_main():
        statistical_results = run_statistical_analysis(all_metrics)
        with open(results_file, "w") as f:
            json.dump(statistical_results, f, indent=2)
        print(f"Saved evaluation results to {results_file}")

    if getattr(args, "distributed", False):
        cleanup()


if __name__ == "__main__":
    main()
