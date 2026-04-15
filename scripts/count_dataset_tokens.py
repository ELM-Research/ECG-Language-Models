import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer
from src.configs.constants import HF_DATASETS, HF_LLMS

p=argparse.ArgumentParser();p.add_argument("--llm",default="llama-3.2-3b-instruct");p.add_argument("--phase",default="sft");p.add_argument("--split",default="fold0_train");p.add_argument("--datasets",nargs="*",default=[]);a=p.parse_args()
tok=AutoTokenizer.from_pretrained(HF_LLMS[a.llm]["tokenizer"]);phase="sft" if a.phase=="sf" else a.phase;all_total=0
for n in (a.datasets or HF_DATASETS):
    tot=0
    for x in load_dataset(f"willxxy/{n}",split=a.split)["text"]:
        try:x=json.loads(x)
        except:pass
        if phase=="pretrain":tot+=len(tok.encode(x if isinstance(x,str) else str(x),add_special_tokens=False))
        else:tot+=sum(len(tok.encode((t.get("value") or t.get("content") or "") if isinstance(t,dict) else str(t),add_special_tokens=False)) for t in (x if isinstance(x,list) else [x]))
    all_total+=tot;print(f"{n}\t{tot}")
print(f"TOTAL\t{all_total}")
