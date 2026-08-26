from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base")
tokenizer.add_tokens(["<ecg>"])
messages = [{"role": "user", "content": "<ecg>" * 100 + "\nInterpret this ECG."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
tokens = tokenizer.all_special_tokens + ["<ecg>"]
print("SPECIAL TOKENS:", dict(zip(tokens, tokenizer.convert_tokens_to_ids(tokens))))
print("\nRENDERED PROMPT:\n", prompt)