class Signal:
    def __call__(self, ecg_input):
        return {"ecg" : ecg_input}