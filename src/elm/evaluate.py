import copy
import gc
import json
from pathlib import Path
import torch
from elm.config.load import get_config
from elm.data.build import build_data
from elm.evaluation.evaluator import evaluate, run_statistical_analysis, save_run
from elm.model import build_model
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
    config, _ = get_config()
    folds, seeds = config["evaluation"]["folds"], config["evaluation"]["seeds"]
    dataset_key = "_".join(name.strip("/").replace("/", "_")
                           for name in config["data"]["data_names"])

    run_summaries = []
    output_dirs = set()
    for fold in folds:
        current = fold_config(config, fold)
        checkpoint = current["model"].get("checkpoint")
        if checkpoint:
            base_dir = Path(checkpoint)
        else:
            base_dir = Path(current["evaluation"]["output_dir"]) / "zero_shot"
        run_dir = base_dir / dataset_key
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.add(run_dir)
        set_seed(seeds[0])
        tokenizer, dataset = build_data(current, training=False)
        model = setup_model(build_model(current, tokenizer), config["gpu"])

        for seed in seeds:
            print(f"Evaluating fold {fold} with seed {seed}")
            set_seed(seed)
            result = evaluate(model, dataset, tokenizer, current)
            path = save_run(result, run_dir, fold, seed)
            run_summaries.append({
                "fold": fold, "seed": seed, "num_pairs": result["num_pairs"],
                "metrics": result["metrics"], "file": path.name,
            })

        del model, dataset, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"runs": run_summaries, "aggregate": run_statistical_analysis(run_summaries)}
    for output_dir in output_dirs:
        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as file:
            json.dump(summary, file, indent=2)
        print(f"Saved evaluation results to {summary_path}")


if __name__ == "__main__":
    main()
