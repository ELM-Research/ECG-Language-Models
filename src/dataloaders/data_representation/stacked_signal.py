import numpy as np
from PIL import Image

from dataloaders.data_representation.base import Base

class StackedSignal(Base):
    def __init__(self, data, llm_tokenizer_components, encoder_tokenizer_components, args):
        super().__init__(data, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"] if llm_tokenizer_components else None
        self.encoder_tokenizer = encoder_tokenizer_components["encoder_tokenizer"]

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
        ecg_stacked_signal = self.signal_to_stacked_signal(ecg_signal)

        if self.encoder_tokenizer is not None:
            diagnostic_report = ecg_np_file["report"]
            if self.args.encoder == "clip-vit-base-patch32":
                encoder_tokenizer_out = self.prepare_clip_input(diagnostic_report, ecg_stacked_signal)
            elif self.args.encoder == "siglip2-so400m-patch16-naflex":
                encoder_tokenizer_out = self.prepare_siglip_input(diagnostic_report, ecg_stacked_signal)
            elif self.args.encoder == "vit-base-patch16-224-in21k":
                encoder_tokenizer_out = self.prepare_vit_input(ecg_stacked_signal)
        else:
            encoder_tokenizer_out = {"ecg_signal": ecg_signal}

        return self.prepare_signal_inputs(instance["text"], encoder_tokenizer_out)

    ### SIGNAL TO STACKED SIGNAL FUNCTIONS ###
    def signal_to_stacked_signal(self, signal):
        normalized_signal, _ = self.normalize(signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(rgb_norm_signal)
