from elm.utils.constants import ROLES, TAG_RE, IMAGE_WORD_RE, LEADING_PREFIX_RE

def clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = IMAGE_WORD_RE.sub(lambda match: "Signal" if match[1][0].isupper() else "signal", text)
    return LEADING_PREFIX_RE.sub("", text)

def normalize_text(messages: list[dict]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = next((message[key] for key in ("role", "from") if key in message), None)
        content = next((message[key] for key in ("content", "value") if key in message), None)
        if not isinstance(role, str) or role.strip().lower() not in ROLES:
            raise ValueError(f"Unknown or missing message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Missing text content in message: {message!r}")
        normalized.append({"role": ROLES[role.strip().lower()], "content": clean_text(content)})
    return normalized

class Text:
    def __init__(self, llm_tokenizer, truncation_length, training_stage = None,
                 enable_thinking = False):
        self.llm_tokenizer = llm_tokenizer
        self.truncation_length = truncation_length
        self.training_stage = training_stage
        self.enable_thinking = enable_thinking
        print(self.enable_thinking)

    def __call__(self, text, ecg_token_placeholders):
        if self.training_stage:
            if self.training_stage == "pretrain":
                return self.prepare_pretrain(text, ecg_token_placeholders)
            elif self.training_stage == "sft":
                 return self.prepare_sft(text, ecg_token_placeholders)
            elif self.training_stage == "rl":
                 return self.prepare_rl(text, ecg_token_placeholders)
        return self.prepare_inference(text)

    def prepare_pretrain(self, text, ecg_token_placeholders):
        # Qwen3/3.5 does not put any special tokens except
        # <|endoftext|> at end and as pad
        return self.llm_tokenizer(f"{ecg_token_placeholders}{text}", truncation = True,
                                  max_length = self.truncation_length,)

    def prepare_sft(self, text, ecg_token_placeholders):
        normalized_text = normalize_text(text)
        user = next((turn for turn in normalized_text if turn["role"] == "user"), None)
        user["content"] = ecg_token_placeholders + user["content"]
        tokenized_text = self.llm_tokenizer.apply_chat_template(
            normalized_text, tokenize = True, truncation = True,
            max_length = self.truncation_length, return_dict = True,
            add_generation_prompt = False,
        )
        tokenized_text["labels"] = tokenized_text["input_ids"].copy()
        return tokenized_text

    def prepare_rl(self, text, ecg_token_placeholders):
        tokenized_text = self.prepare_sft(text, ecg_token_placeholders)

    def prepare_inference(self, text):
        pass