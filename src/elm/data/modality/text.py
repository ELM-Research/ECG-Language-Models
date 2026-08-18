from elm.utils.constants import (
    ECG_TOKEN_PLACEHOLDER,
    IMAGE_WORD_RE,
    LEADING_PREFIX_RE,
    ROLES,
    TAG_RE,
)


def clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = IMAGE_WORD_RE.sub(lambda match: "Signal" if match[1][0].isupper() else "signal", text)
    return LEADING_PREFIX_RE.sub("", text)


def normalize_text(text: list[dict], system_prompt: str = None) -> list[dict[str, str]]:
    normalized = []
    if system_prompt:
        normalized.append({"role": "system", "content": system_prompt})
    for message in text:
        role = next((message[key] for key in ("role", "from") if key in message), None)
        content = next((message[key] for key in ("content", "value") if key in message), None)
        if not isinstance(role, str) or role.strip().lower() not in ROLES:
            raise ValueError(f"Unknown or missing message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Missing text content in message: {message!r}")
        normalized.append({"role": ROLES[role.strip().lower()], "content": clean_text(content)})
    return normalized


class Text:
    def __init__(self, llm_tokenizer, truncation_length, training_stage,
                 system_prompt_path=None):
        self.llm_tokenizer = llm_tokenizer
        self.truncation_length = truncation_length
        self.training_stage = training_stage
        self.system_prompt = None
        if system_prompt_path:
            with open(system_prompt_path, encoding="utf-8") as file:
                self.system_prompt = file.read()

    def __call__(self, text, ecg_token_placeholders):
        if self.training_stage == "pretrain":
            return self.prepare_pretrain(text, ecg_token_placeholders)
        if self.training_stage in ("sft", "rl"):
            return self.prepare_sft(text, ecg_token_placeholders)
        raise ValueError(f"Unknown training stage: {self.training_stage}")

    def prepare_pretrain(self, text, ecg_token_placeholders):
        # Qwen 3.5 pretraining documents have no BOS or EOS. The pad token is
        # the document separator, so it is part of the sequence and its loss.
        input_ids = self.llm_tokenizer.encode(
            f"{ecg_token_placeholders}{text}", add_special_tokens=False,
        )[:self.truncation_length - 1]
        input_ids.append(self.llm_tokenizer.pad_token_id)
        ecg_token = self.llm_tokenizer.convert_tokens_to_ids(ECG_TOKEN_PLACEHOLDER)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100 if token == ecg_token else token for token in input_ids],
        }

    def prepare_sft(self, text, ecg_token_placeholders):
        # Qwen3/3.5 always includes think start and end tokens
        # Even when no thinking content present (content inside think start and end is empty)
        # For our text that already has thinking and answer tags, apply_chat_template
        # automatically allocates the thinking content appropriately.
        normalized_text = normalize_text(text, self.system_prompt)
        user = next((turn for turn in normalized_text if turn["role"] == "user"), None)
        user["content"] = ecg_token_placeholders + user["content"]
        tokenized_text = self.llm_tokenizer.apply_chat_template(
            normalized_text,
            tokenize=True,
            truncation=True,
            max_length=self.truncation_length,
            return_dict=True,
            add_generation_prompt=False,
        )
        input_ids = tokenized_text["input_ids"]
        assistant_header = self.llm_tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False)
        im_end = self.llm_tokenizer.convert_tokens_to_ids("<|im_end|>")
        labels = [-100] * len(input_ids)
        for response_start in (i + len(assistant_header) for i in range(len(input_ids))
                               if input_ids[i:i + len(assistant_header)] == assistant_header):
            response_end = next((i + 1 for i in range(response_start, len(input_ids))
                                 if input_ids[i] == im_end), len(input_ids))
            labels[response_start:response_end] = input_ids[response_start:response_end]
        tokenized_text["labels"] = labels
        return tokenized_text
