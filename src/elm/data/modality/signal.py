from elm.utils.constants import ECG_TOKEN_PLACEHOLDER

class Signal:
    def __init__(self, num_ecg_tokens: int, only_text=False):
        self.num_ecg_tokens = num_ecg_tokens
        self.only_text = only_text

    def __call__(self, ecg_input):
        if self.only_text:
            return {}, ""
        placeholders = ECG_TOKEN_PLACEHOLDER * self.num_ecg_tokens + "\n"
        return {"ecg_values": ecg_input}, placeholders
