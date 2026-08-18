import copy
import gc
import json
from pathlib import Path

import torch

from elm.config.load import get_config
from elm.data.build import build_data
from elm.evaluation.evaluator import evaluate, run_statistical_analysis, save_run
from elm.model import build_model
from elm.utils.logging import setup_experiment_folder
from elm.utils.parallelism import setup_model
from elm.utils.seed import set_seed


def fold_config(config: dict, fold) -> dict:
    configured = copy.deepcopy(config)
    evaluation = configured["evaluation"]
    configured["data"]["split_names"] = [name.format(fold=fold) for name in evaluation["split_names"]]
    checkpoint = configured["model"].get("checkpoint")
    if checkpoint:
        configured["model"]["checkpoint"] = checkpoint.format(fold=fold)
    return configured

def main() -> None:
    config, experiment_name = get_config()
    folds, seeds = config["evaluation"]["folds"], config["evaluation"]["seeds"]
    run_dir = setup_experiment_folder(
        Path(config["evaluation"]["output_dir"]) / experiment_name, config)

    run_summaries = []
    for fold in folds:
        current = fold_config(config, fold)
        set_seed(seeds[0])
        tokenizer, dataloader = build_data(current, training=False)
        model = setup_model(build_model(current, tokenizer), None)

        for seed in seeds:
            print(f"Evaluating fold {fold} with seed {seed}")
            set_seed(seed)
            result = evaluate(model, dataloader, tokenizer, current)
            path = save_run(result, run_dir, fold, seed)
            run_summaries.append({
                "fold": fold, "seed": seed, "num_pairs": result["num_pairs"],
                "metrics": result["metrics"], "file": path.name,
            })

        del model, dataloader, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"runs": run_summaries, "aggregate": run_statistical_analysis(run_summaries)}
    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved evaluation results to {summary_path}")


if __name__ == "__main__":
    main()
