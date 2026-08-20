from datasets import load_dataset
import json

def decode_batch(batch: dict) -> dict:
    if "text" in batch:
        out = []
        for t in batch["text"]:
            try:
                out.append(json.loads(t))
            except Exception:
                out.append(t)
        batch["text"] = out
    return batch

if __name__ == "__main__":
    df = load_dataset("ELM-Research/pretrain-stage1", split = "fold1_train").with_transform(decode_batch).shuffle()
    for i in df:
        print(i)
        input()