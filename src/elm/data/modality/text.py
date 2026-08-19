from elm.utils.constants import (
    IMAGE_WORD_RE,
    LEADING_PREFIX_RE,
    ROLES,
    TAG_RE,
    THINK_START,
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
        normalized.append({"role": ROLES[role.strip().lower()], "content": clean_text(content)})
    return normalized


def chat_prompt(tokenizer, messages: list[dict], explicit_thinking: bool) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    think_prompt = f"{THINK_START}\n"
    if not explicit_thinking:
        prompt = prompt[:-len(think_prompt)]
    return prompt


class Text:
    def __init__(self, llm_tokenizer, truncation_length, training_stage,
                 system_prompt_path=None, truncate=True, explicit_thinking=False):
        self.llm_tokenizer = llm_tokenizer
        self.truncation_length = truncation_length
        self.training_stage = training_stage
        self.truncate = truncate
        self.explicit_thinking = explicit_thinking
        self.system_prompt = None
        if system_prompt_path:
            with open(system_prompt_path, encoding="utf-8") as file:
                self.system_prompt = file.read().strip()

    def __call__(self, text, ecg_token_placeholders):
        if self.training_stage is None:
            return self.prepare_evaluation(text, ecg_token_placeholders)
        if self.training_stage == "pretrain":
            return self.prepare_pretrain(text, ecg_token_placeholders)
        if self.training_stage in ("sft", "rl"):
            return self.prepare_sft(text, ecg_token_placeholders)
        raise ValueError(f"Unknown training stage: {self.training_stage}")

    def prepare_evaluation(self, text, ecg_token_placeholders):
        if isinstance(text, str):
            return {"prompt": ecg_token_placeholders, "reference": text}
        messages = normalize_text(text, self.system_prompt)
        user = next((message for message in messages if message["role"] == "user"), None)
        user["content"] = ecg_token_placeholders + user["content"]
        return {"messages": messages}

    def prepare_pretrain(self, text, ecg_token_placeholders):
        # Qwen 3.5 pretraining documents have no BOS or EOS. The pad token is
        # the document separator, so it is part of the sequence and its loss.
        prompt_ids = self.llm_tokenizer.encode(
            ecg_token_placeholders, add_special_tokens=False)
        response_ids = self.llm_tokenizer.encode(text, add_special_tokens=False)
        if self.truncate:
            available = self.truncation_length - len(prompt_ids) - 1
            response_ids = response_ids[:available]
        separator = self.llm_tokenizer.pad_token_id
        return {
            "input_ids": prompt_ids + response_ids + [separator],
            "attention_mask": [1] * (len(prompt_ids) + len(response_ids) + 1),
            "labels": [-100] * len(prompt_ids) + response_ids + [separator],
        }

    def prepare_sft(self, text, ecg_token_placeholders):
        # The tokenizer's chat template owns all thinking-tag serialization.
        normalized_text = normalize_text(text, self.system_prompt)
        user = next((turn for turn in normalized_text if turn["role"] == "user"), None)
        user["content"] = ecg_token_placeholders + user["content"]

        prompt = chat_prompt(
            self.llm_tokenizer, normalized_text[:-1], self.explicit_thinking)

        conversation = self.llm_tokenizer.apply_chat_template(
            normalized_text, tokenize=False, add_generation_prompt=False,
        )
        prompt_ids = self.llm_tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.llm_tokenizer.encode(
            conversation[len(prompt):], add_special_tokens=False)
        im_end = self.llm_tokenizer.convert_tokens_to_ids("<|im_end|>")
        response_ids = response_ids[:response_ids.index(im_end) + 1]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        if self.truncate:
            input_ids = input_ids[:self.truncation_length]
            labels = labels[:self.truncation_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
