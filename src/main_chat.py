import torch
import gc

from dataloaders.build_dataloader import BuildDataLoader
from dataloaders.data_representation.base import Base

from utils.dir_file_manager import DirFileManager

from elms.build_elm import BuildELM

from utils.gpu_manager import GPUSetup

from configs.config import get_args

def main():
    mode = "inference"
    args = get_args(mode)
    args.mode = mode

    gc.collect()
    torch.cuda.empty_cache()

    build_tokenizer = BuildDataLoader(args)
    llm_tokenizer = build_tokenizer.dataset_mixer.build_llm_tokenizer()

    build_elm = BuildELM(args)
    elm_components = build_elm.build_elm(llm_tokenizer["llm_tokenizer"])

    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(elm_components["elm"], False)
    device = next(elm.parameters()).device

    chat = Base(None, args)
    chat.llm_tokenizer = llm_tokenizer["llm_tokenizer"]

    print("Input an ECG path")
    ecg_path = input()
    ecg_file = DirFileManager.open_npy(ecg_path)
    ecg = ecg_file["ecg"]
    normalized_ecg, _ = chat.normalize(ecg)
    ecg_tensor = torch.unsqueeze(torch.tensor(normalized_ecg), dim=0).to(device)

    formatted_input = []

    elm.eval()
    while True:
        user_input = input("User: ")
        formatted_input.append({"role": "user", "content": user_input})
        formatted_input.append({"role": "assistant", "content": ""})
        formatted_prompt = chat.make_prompt(formatted_input)
        tokenized_prompt = chat.prepare_input_ids(formatted_prompt)
        elm_input_ids = torch.unsqueeze(torch.tensor(tokenized_prompt), dim=0).to(device)

        encoder_tokenizer_out = {"ecg_signal": ecg_tensor.to(device)}

        elm_attention_mask = torch.ones(elm_input_ids.shape).to(device)

        signal_id_indices = torch.unsqueeze(torch.tensor(chat.find_signal_token_indices(tokenized_prompt)), dim=0).to(device)

        generate_output = elm.generate(elm_input_ids,
         encoder_tokenizer_out,
         elm_attention_mask,
         signal_id_indices,
         args.max_new_tokens)
        
        response = chat.decode_response(generate_output[0].tolist())
        print(response)

        formatted_input[-1] = {"role": "assistant", "content": response}

if __name__ == "__main__":
    main()
