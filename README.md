<h2 align="center">
  Scaling ECG-Language Model Reasoning to 13 Million Training Examples
</h2>

<div align="center">
  <img src="./assets/orah.png" alt="The Orah Model">
</div>

## Overview <a name="overview"></a>
This is the full training, evaluation, and inferencing code for Orah, an ECG-Language Model scaled to 13 million training examples.
Prepare datasets with [ECG-Preprocess](https://github.com/ELM-Research/ECG-Preprocess) before use. Additionally, if you want to pretrain SigLEP, please view [ECG-Neural-Networks](https://github.com/ELM-Research/ECG-Neural-Networks).

We previously supported multiple ELM architectures, ECG representations, etc., however, we decided to completely dedicate this repository to training and evaluting Orah. We plan to iterate upon Orah. All updates will be made in this repository. 
Please feel free to contribute to the repository!
If there are any questions or bugs, please do not hesitate to reach out to wjhan{@}andrew{dot}cmu{edu} or submit an issue with corresponding details.

> **Status:** Beta.

## Setup <a name="setup"></a>
We use torch 2.9.1 with cuda 12.8 and primarily use H100 NVL NVIDIA GPUs.


```bash
git clone https://github.com/ELM-Research/ELM.git
cd ELM && uv sync
```

### Notes

1. To most optimally run Qwen3.5, it is recommended to install `causal-conv1d` and `flash-linear-attention`. We include it in the pyproject.toml file as a default install. However, if one has trouble installing it, please refer to their respective repos, or feel free to ignore the install. Ignoring will simply default Qwen3.5 to less optimized kernels.

## Training Datasets <a name="data"></a>

First, preprocess the ECGs using the [ECG-Preprocess](https://github.com/ELM-Research/ECG-Preprocess) repository.


| Stage              | Trainable Modules | Epochs | Dataset                                                                                  | Hugging Face Dataset |        Samples |
| ------------------ | ----------------- | -----: | ---------------------------------------------------------------------------------------- | -------------------- | -------------: |
| SigLEP Pretraining | Encoder           |     30 | [Harvard-Emory ECG Database (HEEDB)](https://bdsp.io/content/heedb/)                                                         |                      |      1,927,353 |
|                    |                   |        | **Total**                                                                                |                      |  **1,927,353** |
| Orah Pretraining 1 | Connector         |     3 | [EchoNext](https://physionet.org/content/echonext/1.1.1/)                                                |                      |         24,763 |
|                    |                   |        | [Harvard-Emory ECG Database (HEEDB)](https://bdsp.io/content/heedb/)                                                         |                      |        642,451 |
|                    |                   |        | **Total**                                                                                |                      |    **667,214** |
| Orah Pretraining 2 | Connector, LLM    |      1 | [EchoNext](https://physionet.org/content/echonext/1.1.1/)                                                |                      |         57,780 |
|                    |                   |        | [Harvard-Emory ECG Database (HEEDB)](https://bdsp.io/content/heedb/)                                                         |                      |      5,996,208 |
|                    |                   |        | **Total**                                                                                |                      |  **6,053,988** |
| Orah SFT 1         | Connector, LLM    |      3 | [ECG-QA MIMIC-IV](https://github.com/Jwoo5/ecg-qa)                        |                      |        352,382 |
|                    |                   |        | [Pretrain MIMIC](https://github.com/YubaoZhao/ECG-Chat)                              |                      |        502,687 |
|                    |                   |        | [ECG-Instruct 45K](https://github.com/YubaoZhao/ECG-Chat)                            |                      |         44,778 |
|                    |                   |        | [ECG-Grounding](https://github.com/lanxiang1017/GEM)                            |                      |        353,210 |
|                    |                   |        | **Total**                                                                                |                      |  **1,253,057** |
| Orah SFT 2         | Connector, LLM    |      3 | [ECG-QA MIMIC-IV](https://github.com/Jwoo5/ecg-qa)                        |                      |        822,226 |
|                    |                   |        | [ECG-Grounding](https://github.com/lanxiang1017/GEM)                                   |                      |        824,158 |
|                    |                   |        | [ECG-Instruct ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1)               |                      |      1,147,368 |
|                    |                   |        | [ECG Protocol-Guided Grounding CoT](https://github.com/PKUDigitalHealth/ECG-R1) |                      |         30,000 |
|                    |                   |        | [ECG-QA-CoT](https://github.com/OpenTSLM/OpenTSLM/tree/main)                            |                      |        159,313 |
|                    |                   |        | **Total**                                                                                |                      |  **2,983,065** |
| Orah RL            | Connector, LLM    |      3 | [RL ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1)                         |                      |          3,948 |
|                    |                   |        | **Total**                                                                                |                      |      **3,948** |
|                    |                   |        | **Grand Total**                                                                          |                      | **12,888,625** |


## Contributions <a name="contributions"></a>

We welcome contributions to the repository! Please feel free to open an issue or pull request for any bugs or features you would like to add. We are always looking for new ECG datasets to benchmark our methods on. If you have any recommendations, please let us know!

For most processes, we have a `--dev` flag to run in a smaller scale and add some verbosity for debugging. Feel free to add this flag when needed!

## Acknowledgements <a name="ack"></a>
This work is done in collaboration with the Mario Lemieux Center for Heart Rhythm Care at Allegheny General Hospital.

We thank Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao, Hyoeun Kang, Wenhao Ding, Haohong Lin, Shiqi Liu, Xiaoyu (Simon) Song, Tony Chen, Atharva Mhaskar, Zhepeng Cen, Yihang Yao, and Dylan Leong for their helpful discussions, feedbacks, and support in developing the initial [ECG-Bench](https://github.com/willxxy/ECG-Bench) which turned into the current ELM repository.

We thank the authors of [ECG-Byte](https://github.com/willxxy/ECG-Byte), [MERL](https://github.com/cheliu-computation/MERL-ICML2024), [ST-MEM](https://github.com/bakqui/ST-MEM), [ECG-QA](https://github.com/Jwoo5/ecg-qa), [ECG-Chat](https://github.com/YubaoZhao/ECG-Chat), [PULSE](https://github.com/AIMedLab/PULSE), [GEM](https://github.com/lanxiang1017/GEM), [ECG-R1](https://github.com/PKUDigitalHealth/ECG-R1) for their code and publicly released datasets.

Lastly, we thank [HuggingFace](https://huggingface.co/) for providing the APIs for the models.

## License

MIT, except all third-party libraries, models, and datasets used in the repository. Please refer to the third-party library, model and dataset's corresponding licenses.
