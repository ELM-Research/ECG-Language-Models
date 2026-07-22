from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from tqdm import tqdm
print("flash attention SDP enabled.", torch.backends.cuda.flash_sdp_enabled())

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

messages = [
    {"role": "user", "content": "Who are you?"},
]

attn_impls = ["kernels-community/flash-attn2", "kernels-community/flash-attn3",
              "kernels-community/vllm-flash-attn3", "sdpa", None]
warmup_runs = 10
timed_runs = 100

for impl in attn_impls:
    print(f"\n{'='*50}")
    print(f"Attention: {impl}")
    print(f"{'='*50}")

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        attn_implementation=impl,
    )
    model = torch.compile(model)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    for _ in tqdm(range(warmup_runs), desc = f"Warming up {impl}"):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=40, pad_token_id = tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in tqdm(range(timed_runs), desc = f"Running {impl}"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id = tokenizer.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"Avg over {timed_runs} runs: {avg:.4f}s")
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

    del model
    torch.cuda.empty_cache()

'''
kernels-community/flash-attn2
Avg over 100 runs: 0.5858s

kernels-community/flash-attn3
Avg over 100 runs: 0.6026s

kernels-community/vllm-flash-attn3
Avg over 100 runs: 0.5615s

sdpa
Avg over 100 runs: 0.4683s

None
Avg over 100 runs: 0.4665s
'''