from elm.utils.constants import ECG_TOKEN_PLACEHOLDER

class Signal:
    def __init__(self, ecg_tokens, only_text = False):
        self.ecg_tokens = ecg_tokens
        self.only_text = only_text

    def __call__(self, ecg_input):
        if self.only_text:
            return {}, ""
        if self.ecg_tokens["mode"] != "static":
            raise ValueError(f"Unknown ECG token mode: {self.ecg_tokens['mode']}")
        return {"ecg_values": ecg_input}, self.prepare_static_ecg_tokens()

    def prepare_static_ecg_tokens(self,):
        return ECG_TOKEN_PLACEHOLDER*self.ecg_tokens["num_ecg_tokens"] + "\n"
