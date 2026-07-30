from elm.utils.constants import ECG_TOKEN_PLACEHOLDER

class Signal:
    def __init__(self, ecg_tokens):
        self.ecg_tokens = ecg_tokens

    def __call__(self, ecg_input):

        return {"ecg" : ecg_input,}, \
            self.prepare_static_ecg_tokens() if self.ecg_tokens["mode"] else self.prepare_dynamic_ecg_tokens()

    def prepare_static_ecg_tokens(self,):
        return ECG_TOKEN_PLACEHOLDER*self.ecg_tokens["num_ecg_tokens"]

    def prepare_dynamic_ecg_tokens(self,):
        pass