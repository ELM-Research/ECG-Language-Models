<h2 align="center">
  A Training and Evaluation Framework for ECG-Language Models (ELMs)
</h2>

<div align="center">
  <img src="./assets/fig1_2.png" alt="Our pipeline.">
</div>

## Overview <a name="overview"></a>
A research framework for finetuning and evaluating ECG-language models (ELMs). Supports multiple architectures, training objectives, and data representations with distributed training out of the box.
Prepare datasets with [ECG-Preprocess](https://github.com/ELM-Research/ECG-Preprocess) before use. Additionally, if you want to pretrain an ECG encoder, please view [ECG-Neural-Networks](https://github.com/ELM-Research/ECG-Neural-Networks).

We hope to continuously update the repository to support more features, ELMs, and datasets. Please feel free to contribute to the repository!
If there are any questions or bugs, please do not hesitate to reach out to wjhan{@}andrew{dot}cmu{edu} or submit an issue with corresponding details.

> **Status:** Beta.

## Setup <a name="setup"></a>
We use torch 2.9 with cuda 12.8 and primarily use H100 NVL NVIDIA GPUs.


```bash
git clone https://github.com/ELM-Research/ELM.git
cd ELM && uv sync
```

For BPE symbolic representation with [ECG-Byte](https://arxiv.org/abs/2412.14373), do:

```bash
uv sync --extra ecg_byte
```

If Rust is not installed run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.82.0 -y`, open a new terminal to set the PATH, and rerun the above commands.

## ECG Datasets <a name="data"></a>

First, preprocess the ECGs using the [ecg_preprocess](https://github.com/ELM-Research/ecg_preprocess) repository.
The structure in which the `data` folder should be in is the following:

```
data
├── csn
│   ├── preprocessed_1250
│   ├── preprocessed_500
│   └── preprocessed_2500
├── cpsc
│   └── ...
├── ptb_xl
│   └── ...
├── mimic_iv
│   └── ...
└── code15
    └── ...
```

We support the following datasets in a unified way through datasets from HuggingFace. These datasets will include the `ecg_path` which is the path to the `.npy` files in the `data` folder. It will also include the conversational data (`text`).

| `--data`  | Link        |
|----------|------------|
| [ECG-QA PTB-XL](https://arxiv.org/abs/2306.15681)  | [ELM-Research/ecg-qa-ptbxl-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-qa-ptbxl-250-2500)   |
| [ECG-QA MIMIC-IV-ECG](https://arxiv.org/abs/2306.15681) | [ELM-Research/ecg-qa-mimic-iv-ecg-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-qa-mimic-iv-ecg-250-2500) |
| [Pretrain Mimic](https://arxiv.org/abs/2408.08849)  | [ELM-Research/pretrain-mimic-250-2500](https://huggingface.co/datasets/ELM-Research/pretrain-mimic-250-2500)   |
| [ECG-Grounding](https://www.arxiv.org/abs/2503.06073)    | [ELM-Research/ecg-grounding-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-grounding-250-2500)     |
| [ECG-Instruct Pulse](https://arxiv.org/abs/2410.19008)     | [ELM-Research/ecg-instruct-pulse-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-instruct-pulse-250-2500)      |
| [ECG-Bench Pulse](https://arxiv.org/abs/2410.19008)     | [ELM-Research/ecg-bench-pulse-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-bench-pulse-250-2500)      |
| [ECG-Instruct 45k](https://arxiv.org/abs/2408.08849)     | [ELM-Research/ecg-instruct-45k-250-2500](https://huggingface.co/datasets/ELM-Research/ecg-instruct-45k-250-2500)      |
| [ECG-QA-CoT](https://github.com/StanfordBDHG/OpenTSLM/tree/main)     | [ELM-Research/ecg-qa-cot](https://huggingface.co/datasets/ELM-Research/ecg-qa-cot)      |
| [ECG-Protocol-Guided-Grounding-CoT RL](https://huggingface.co/datasets/PKUDigitalHealth/ECG-Protocol-Guided-Grounding-CoT/viewer/rl)     | [ELM-Research/rl-ecg-r1](https://huggingface.co/datasets/ELM-Research/rl-ecg-r1)    
| [ECG-Protocol-Guided-Grounding-CoT Base](https://huggingface.co/datasets/PKUDigitalHealth/ECG-Protocol-Guided-Grounding-CoT/viewer/base)     | [ELM-Research/base-ecg-r1](https://huggingface.co/datasets/ELM-Research/base-ecg-r1)      |


## ECG Representations <a name="representation"></a>

| `--data_representation` | Description |
|-------------------------|-------------|
| `signal` | Raw ECG matrix $X \in \mathbb{R}^{C \times L}$ (leads × samples) |

## ELMs
We implement several ELMs and describe how to train each variant.


## Research
We list research projects that have been conducted using this repository. Please feel free to add your own research here!

- [ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling
](https://arxiv.org/abs/2412.14373)
- [Signal, Image, or Symbolic: Exploring the Best Input Representation for Electrocardiogram-Language Models Through a Unified Framework](https://arxiv.org/abs/2505.18847)
- [Retrieval-Augmented Generation for Electrocardiogram-Language Models](https://arxiv.org/abs/2510.00261)
- [ELF: A Family of Encoder-Free ECG-Language Models](https://arxiv.org/abs/2601.18798)

## Contributions <a name="contributions"></a>

We welcome contributions to the repository! Please feel free to open an issue or pull request for any bugs or features you would like to add. We are always looking for new ECG datasets to benchmark our methods on. If you have any recommendations, please let us know!

For most processes, we have a `--dev` flag to run in a smaller scale and add some verbosity for debugging. Feel free to add this flag when needed!

## Acknowledgements <a name="ack"></a>
This work is done in collaboration with the Mario Lemieux Center for Heart Rhythm Care at Allegheny General Hospital.

We thank Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao, Hyoeun Kang, Wenhao Ding, Haohong Lin, Shiqi Liu, Xiaoyu (Simon) Song, Tony Chen, Atharva Mhaskar, Zhepeng Cen, Yihang Yao, and Dylan Leong for their helpful discussions, feedbacks, and support in developing the initial [ECG-Bench](https://github.com/willxxy/ECG-Bench) which turned into the current ELM repository.

We thank the authors of [ECG-Byte](https://github.com/willxxy/ECG-Byte), [MERL](https://github.com/cheliu-computation/MERL-ICML2024), [ST-MEM](https://github.com/bakqui/ST-MEM), [ECG-QA](https://github.com/Jwoo5/ecg-qa), [ECG-Chat](https://github.com/YubaoZhao/ECG-Chat), [PULSE](https://github.com/AIMedLab/PULSE), [GEM](https://github.com/lanxiang1017/GEM), [ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1) for their code and publicly released datasets.

Lastly, we thank [HuggingFace](https://huggingface.co/) for providing the APIs for the models.

## License

MIT, except all third-party models and datasets used in the repository. Please refer to the third-party model and dataset's corresponding licenses.
