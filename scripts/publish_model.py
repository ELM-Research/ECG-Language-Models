import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("output")
parser.add_argument("--tokenizer")
parser.add_argument("--repo-id")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
hub = {"push_to_hub": True, "repo_id": args.repo_id} if args.repo_id else {}
model.save_pretrained(args.output, **hub)
tokenizer.save_pretrained(args.output, **hub)