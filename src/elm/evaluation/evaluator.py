import json
import string
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm
from elm.data.modality.text import chat_prompt

ROUGE_SCORER = RougeScorer(["rougeL"], use_stemmer=True)

def split_response(text: str, explicit_thinking: bool = False) -> tuple[str, str]:
    thinking = ""
    in_thinking = explicit_thinking
    if "<think>" in text:
        _, _, text = text.partition("<think>")
        in_thinking = True
    if in_thinking or "</think>" in text:
        thinking, closed, text = text.partition("</think>")
        if not closed:
            return thinking.strip(), ""

    if "<answer>" in text:
        _, _, text = text.partition("<answer>")
    if "</answer>" in text:
        text, _, _ = text.partition("</answer>")
    return thinking.strip(), text.strip()


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def token_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize(reference).split()
    hypothesis_tokens = normalize(hypothesis).split()
    if not reference_tokens or not hypothesis_tokens:
        return float(reference_tokens == hypothesis_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(hypothesis_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_strings(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    pairs = list(zip(references, hypotheses))
    if not pairs:
        return {name: 0.0 for name in ("accuracy", "f1", "bleu_4", "rouge_l", "meteor")}
    references, hypotheses = map(list, zip(*pairs))
    return {
        "accuracy": float(np.mean([reference == hypothesis for reference, hypothesis in pairs])),
        "f1": float(np.mean([token_f1(reference, hypothesis) for reference, hypothesis in pairs])),
        "bleu_4": float(corpus_bleu(
            [[reference.split()] for reference in references],
            [hypothesis.split() for hypothesis in hypotheses],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )),
        "rouge_l": float(np.mean([
            ROUGE_SCORER.score(reference, hypothesis)["rougeL"].fmeasure
            for reference, hypothesis in pairs
        ])),
        "meteor": float(np.mean([
            meteor_score([reference.split()], hypothesis.split())
            for reference, hypothesis in pairs
        ])),
    }


def eos_ids(model, tokenizer) -> list[int]:
    generation_model = getattr(model, "language_model", model)
    configured = getattr(getattr(generation_model, "generation_config", None), "eos_token_id", None)
    ids = [configured] if isinstance(configured, int) else list(configured or [])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.append(im_end)
    return sorted(set(ids))


def generate_response(model, input_ids: list[int], ecg_values, tokenizer, evaluation: dict) -> str:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    stop_ids = eos_ids(model, tokenizer)
    generation = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": evaluation["max_new_tokens"],
        "do_sample": evaluation["do_sample"],
        "use_cache": True,
    }
    if ecg_values is not None:
        generation["ecg_values"] = torch.as_tensor(ecg_values, device=device).unsqueeze(0)
    if stop_ids:
        generation["eos_token_id"] = stop_ids
    if tokenizer.pad_token_id is not None:
        generation["pad_token_id"] = tokenizer.pad_token_id
    if evaluation["do_sample"]:
        generation["temperature"] = evaluation.get("temperature", 1.0)

    output = model.generate(**generation)[0]
    prompt_length = input_ids.shape[1]
    if output.shape[0] >= prompt_length and torch.equal(output[:prompt_length], input_ids[0]):
        output = output[prompt_length:]
    stop = next((index for index, token in enumerate(output.tolist()) if token in stop_ids), len(output))
    return tokenizer.decode(output[:stop], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False).strip()

def print_generation(example: int, turn: int, prompt: str, reference: str,
                     hypothesis: str, explicit_thinking: bool) -> None:
    print(f"\n=== Evaluation formulation (example {example}, turn {turn}) ===")
    print(f"explicit_thinking: {explicit_thinking}")
    print("[Prompt]")
    print(prompt)
    print("[Reference]")
    print(reference)
    print("[Generated response]")
    print(hypothesis)


def evaluate(model, dataset, tokenizer, config: dict) -> dict:
    model.eval()
    records = []
    explicit_thinking = config.get("explicit_thinking", False)
    progress = tqdm(dataset, desc="Evaluation", leave=False)
    with torch.inference_mode():
        for example_index, example in enumerate(progress):
            ecg_values = example.get("ecg_values")
            if "messages" not in example:
                prompt = example["prompt"]
                reference = example["reference"]
                input_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
                hypothesis = generate_response(
                    model, input_ids, ecg_values, tokenizer, config["evaluation"])
                records.append({
                    "prompt": prompt, "predicted": hypothesis,
                    "ground_truth": reference,
                })
                if config["development"]:
                    print_generation(
                        example_index + 1, 1, prompt, reference, hypothesis, explicit_thinking)
            else:
                history = []
                turn = 0
                for message in example["messages"]:
                    if message["role"] != "assistant":
                        history.append(message)
                        continue

                    turn += 1
                    prompt = chat_prompt(tokenizer, history, explicit_thinking)
                    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    reference = message["content"]
                    hypothesis = generate_response(
                        model, input_ids, ecg_values, tokenizer, config["evaluation"])
                    records.append({
                        "prompt": prompt,
                        "ground_truth": reference,
                        "predicted": hypothesis,
                    })
                    if config["development"]:
                        print_generation(
                            example_index + 1, turn, prompt, reference, hypothesis,
                            explicit_thinking)
                    content = f"<think>\n{hypothesis}" if explicit_thinking else hypothesis
                    history.append({"role": "assistant", "content": content})
            if config["development"] and example_index == 0:
                break

    references = [record["ground_truth"] for record in records]
    hypotheses = [record["predicted"] for record in records]
    reference_parts = [split_response(text) for text in references]
    hypothesis_parts = [split_response(text, explicit_thinking) for text in hypotheses]
    answer_references = [parts[1] for parts in reference_parts]
    answer_hypotheses = [parts[1] for parts in hypothesis_parts]
    metrics = {"answer": evaluate_strings(answer_references, answer_hypotheses)}

    thinking = [(reference[0], hypothesis[0]) for reference, hypothesis in zip(reference_parts, hypothesis_parts)
                if reference[0]]
    if thinking:
        thinking_references, thinking_hypotheses = zip(*thinking)
        metrics["thinking"] = evaluate_strings(list(thinking_references), list(thinking_hypotheses))

    result = {
        "num_pairs": len(records), "metrics": metrics, "predictions": records,
    }
    print_metrics(result)
    return result


def run_statistical_analysis(results: list[dict]) -> dict:
    if not results:
        raise ValueError("At least one evaluation result is required")

    def summarize(values):
        if any(isinstance(value, Mapping) for value in values):
            keys = sorted({key for value in values if isinstance(value, Mapping) for key in value})
            return {key: summarize([value[key] for value in values if isinstance(value, Mapping) and key in value])
                    for key in keys}
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        if len(values) == 1:
            return {"mean": mean, "std": 0.0, "confidence_interval": [mean, mean], "n": 1}
        std = float(values.std(ddof=1))
        margin = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        return {"mean": mean, "std": std, "confidence_interval": [mean - margin, mean + margin], "n": len(values)}

    return summarize([result["metrics"] for result in results])


def print_metrics(result: dict) -> None:
    print(f"\nEvaluation pairs: {result['num_pairs']}")
    for group, metrics in result["metrics"].items():
        print(f"[{group}]")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


def save_run(result: dict, output_dir: Path, fold, seed) -> Path:
    path = output_dir / f"fold_{fold}_seed_{seed}.json"
    with path.open("w") as file:
        json.dump(result["predictions"], file, indent=2)
    return path
