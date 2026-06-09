from typing import Optional

from dataloaders.data_representation.base import Base
from utils.gpu_manager import is_main


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
        # print("ecg_signal", ecg_signal.shape)
        encoder_tokenizer_out = {"ecg_signal": self.transform_ecg_signal(ecg_signal)}

        text = instance["text"]
        prompt = self.make_prompt(text)
        if self.args.dev and is_main():
            print("prompt\n", prompt)

        if "train" in self.args.mode:
            return self.prepare_training_set(prompt, encoder_tokenizer_out)
        else:
            return self.prepare_eval_inference_set(prompt, encoder_tokenizer_out)

    def prepare_training_set(
        self,
        prompt: Optional[str],
        encoder_tokenizer_out: dict,
    ):
        item = super().prepare_training_set(prompt, encoder_tokenizer_out)
        assert len(item["signal_id_indices"]) == self.args.num_encoder_tokens
        return item

    def transform_ecg_signal(self, ecg_signal):
        if self.args.elm == "base_elf":
            return ecg_signal.flatten()
        else:
            return ecg_signal
