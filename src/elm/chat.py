
import numpy as np

from elm.config.load import get_config
from elm.data.build import DataBuilder
from elm.data.modality.signal import Signal
from elm.data.modality.text import Text, chat_prompt
from elm.evaluation.evaluator import generate_response
from elm.model import build_model
from elm.utils.parallelism import setup_model
from elm.utils.seed import set_seed


def main() -> None:
    config, _ = get_config()
    set_seed(config["seed"])
    explicit_thinking = config["explicit_thinking"]
    tokenizer = DataBuilder(config, training=False).build_llm_tokenizer()
    model = setup_model(build_model(config, tokenizer), config["gpu"]).eval()
    text_preparer = Text(tokenizer, config["model"]["truncation_length"], None,
                         system_prompt_path=config["system_prompt_path"])

    opened_npy = np.load(input("ECG path: ").strip(), allow_pickle=True).item()
    ecg = opened_npy["ecg"]
    print("Report", opened_npy["report"])
    ecg_input, placeholders = Signal(config["model"]["num_ecg_tokens"])(ecg)

    history = []
    while True:
        message = input("\nUser: ").strip()
        if message == "break": break
        history.append({"role": "user", "content": message})
        prompt = chat_prompt(tokenizer, text_preparer(history, placeholders)["messages"], explicit_thinking)
        response = generate_response(model, tokenizer.encode(prompt, add_special_tokens=False),
                                     ecg_input["ecg_values"], tokenizer, config["evaluation"])
        print(f"Assistant: {response}")
        history.append({"role": "assistant",
                        "content": f"<think>\n{response}" if explicit_thinking else response})
if __name__ == "__main__":
    main()
