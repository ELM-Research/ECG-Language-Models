

class Text:
    def __init__(self, llm_tokenizer, truncation_length, training_stage = None):
        self.llm_tokenizer = llm_tokenizer
        self.truncation_length = truncation_length
        self.training_stage = training_stage

    def __call__(self, text, ecg_token_placeholders):
        if self.training_stage:
            if self.training_stage == "pretrain":
                return self.prepare_pretrain(text, ecg_token_placeholders)
            elif self.training_stage == "sft":
                 return self.prepare_sft(text)
            elif self.training_stage == "rl":
                 return self.prepare_rl(text)
        return self.prepare_inference(text)

    def prepare_pretrain(self, text, ecg_token_placeholders):
        # Qwen3/3.5 does not put any special tokens except
        # <|endoftext|> at end and as pad
        tokenized_input = self.llm_tokenizer(f"{ecg_token_placeholders}\n{text}",
                                             truncation = True,
                                             max_length = self.truncation_length,)
        return tokenized_input

    def prepare_sft(self, text):
            pass

    def prepare_rl(self, text):
            pass

    def prepare_inference(self, text):
         pass