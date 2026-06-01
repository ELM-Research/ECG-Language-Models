from dataloaders.data_representation.base import Base


class Signal(Base):
    def __init__(self, data, llm_tokenizer_components, args):
        super().__init__(data, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"]

    def __getitem__(self, index):
        instance = self.data[index]
        if instance["ecg_path"] == "noise" or self.args.perturb == "noise":
            ecg_signal = self.gauss_noise_ecg()
        elif instance["ecg_path"] == "flatline" or self.args.perturb == "zeros":
            ecg_signal = self.blackout_ecg()
        else:
            ecg_np_file = self.fm.open_npy(instance["ecg_path"])
            ecg_signal = ecg_np_file["ecg"][self.args.leads]
            if self.args.augment_ecg:
                ecg_signal = self.augment_ecg(ecg_signal)

        ecg_signal, _ = self.normalize(ecg_signal)
        encoder_tokenizer_out = {"ecg_signal": self.transform_ecg_signal(ecg_signal)}
        return self.prepare_signal_inputs(instance["text"], encoder_tokenizer_out)

    def transform_ecg_signal(self, ecg_signal):
        if self.args.elm == "base_elf":
            return ecg_signal.flatten()
        else:
            return ecg_signal
