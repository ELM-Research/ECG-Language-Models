from elm.utils.constants import ECG_TOKEN_PLACEHOLDER

class Signal:
    def __init__(self, ecg_tokens, only_text = False):
        self.ecg_tokens = ecg_tokens
        self.only_text = only_text

    def __call__(self, ecg_input):
        if self.only_text:
            ecg_token_placeholders = ""
        else:
            if self.ecg_tokens["mode"] == "static":
                ecg_token_placeholders = self.prepare_static_ecg_tokens()
            elif self.ecg_tokens["mode"] == "dynamic":
                ecg_token_placeholders = self.prepare_dynamic_ecg_tokens()
        return {"ecg_values": ecg_input}, ecg_token_placeholders

    def prepare_static_ecg_tokens(self,):
        return ECG_TOKEN_PLACEHOLDER*self.ecg_tokens["num_ecg_tokens"] + "\n"

    def prepare_dynamic_ecg_tokens(self,):
        pass
