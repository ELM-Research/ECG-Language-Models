from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoProcessor

from utils.dir_file_manager import DirFileManager
from utils.gpu_manager import is_main

from configs.constants import HF_DATASETS, HF_LLMS, SIGNAL_TOKEN_PLACEHOLDER,\
                                VISION_ENCODERS, ECG_ENCODERS, ECG_TOKEN_PREFIX, RL_TOKENS

class DatasetMixer:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.dfm = DirFileManager()

    def build_torch_dataset(self, ):
        data = []
        for data_name in self.args.data:
            if data_name in HF_DATASETS:
                dataset = self.build_hf_dataset(data_name)
            data.extend(dataset)
        if is_main():
            print(f"Length of Dataset: {len(data)}")
            print(f"Using {self.args.data_representation} representation")
        encoder_tokenizer_components = self.build_encoder_tokenizer()
        llm_tokenizer_components = self.build_llm_tokenizer()
        torch_dataset = self.build_data_representation(data, llm_tokenizer_components,
                                                       encoder_tokenizer_components)
        return torch_dataset

    def build_data_representation(self, data, llm_tokenizer_components,
                                  encoder_tokenizer_components):
        if self.args.data_representation == "signal":
            from dataloaders.data_representation.signal import Signal
            return Signal(data, llm_tokenizer_components, self.args)
        elif self.args.data_representation == "symbolic":
            from dataloaders.data_representation.symbolic import Symbolic
            return Symbolic(data, llm_tokenizer_components, self.args)
        elif self.args.data_representation == "stacked_signal":
            from dataloaders.data_representation.stacked_signal import StackedSignal
            return StackedSignal(data, llm_tokenizer_components,
                                 encoder_tokenizer_components, self.args)
        elif self.args.data_representation == "rgb":
            from dataloaders.data_representation.rgb import RGB
            return RGB(data, llm_tokenizer_components,
                       encoder_tokenizer_components, self.args)

        raise ValueError(f"Unknown data representation: {self.args.data_representation}")

    def build_hf_dataset(self, data_name):
        if self.args.mode in ["train", "post_train"]:
            data = load_dataset(
                f"willxxy/{data_name}",
                split=f"fold{self.args.fold}_train",
            ).with_transform(self.decode_batch)
        elif self.args.mode in ["eval", "inference"]:
            data = load_dataset(
                f"willxxy/{data_name}",
                split=f"fold{self.args.fold}_test",
            ).with_transform(self.decode_batch)
        if self.args.data_subset:
            n = int(len(data) * self.args.data_subset)
            data = data.shuffle(seed=self.args.seed).select(range(n))

        if is_main():
            print("Length of Dataset Considered:", len(data))

        return data

    def decode_batch(self, batch: dict) -> dict:
        if "text" in batch:
            out = []
            for t in batch["text"]:
                try:
                    out.append(json.loads(t))
                except Exception:
                    out.append(t)
            batch["text"] = out
        return batch

    def build_encoder_tokenizer(
        self,
    ):
        if self.args.encoder in VISION_ENCODERS:
            return {"encoder_tokenizer": AutoProcessor.from_pretrained(VISION_ENCODERS[self.args.encoder]["tokenizer"])}
        else:
            return {"encoder_tokenizer": None}

    def build_llm_tokenizer(
        self,
    ):
        llm_tokenizer = AutoTokenizer.from_pretrained(HF_LLMS[self.args.llm]["tokenizer"])
        return self.modify_llm_tokenizer(llm_tokenizer)

    def modify_llm_tokenizer(self, llm_tokenizer):
        if self.args.dev and is_main():
            print("Before Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)

        if getattr(llm_tokenizer, "pad_token", None) is None:  # llama 3.2
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        tokens_to_add = HF_LLMS[self.args.llm]["tokens_to_add"]
        tokens_to_add["additional_special_tokens"].append(SIGNAL_TOKEN_PLACEHOLDER)
        if getattr(self.args, "train_phase", "sft") == "rl":
            tokens_to_add["additional_special_tokens"].extend(RL_TOKENS)
        llm_tokenizer.add_special_tokens(tokens_to_add)

        if self.args.data_representation == "symbolic":
            new_vocab, ecg_byte_builder = self.build_ecg_byte()
            llm_tokenizer.add_tokens(new_vocab)
            out = {"llm_tokenizer": llm_tokenizer, "ecg_tokenizer": ecg_byte_builder}
        else:
            out = {"llm_tokenizer": llm_tokenizer}

        if self.args.dev and is_main():
            print("After Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)
        return out

    def build_ecg_byte(
        self,
    ):
        from dataloaders.data_representation.bpe.ecg_byte import BuildECGByte
        ecg_byte_builder = BuildECGByte(self.args)
        new_vocab = [f"{ECG_TOKEN_PREFIX}{ids!s}" for ids in list(ecg_byte_builder.vocab.keys())]
        if self.args.dev and is_main():
            print("Length of new tokens", len(new_vocab))
        return new_vocab, ecg_byte_builder

    ### DEV FUNCTIONS ###
    def print_llm_tokenizer_info(self, llm_tokenizer):
        print("Vocab Size:", len(llm_tokenizer))
        print("special_tokens_map:", llm_tokenizer.special_tokens_map)
        print("all_special_tokens:", llm_tokenizer.all_special_tokens)
        print("all_special_ids:", llm_tokenizer.all_special_ids)
        for k in ("pad", "bos", "eos", "unk"):
            t = getattr(llm_tokenizer, f"{k}_token", None)
            i = getattr(llm_tokenizer, f"{k}_token_id", None)
            print(f"{k.upper()} -> token: {t!r}, id: {i}")
        print("-" * 20)