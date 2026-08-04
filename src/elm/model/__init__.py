from importlib import import_module


def build_model(config, tokenizer):
    module = import_module(f".{config['model']['name']}", __name__)
    return module.build(config, tokenizer)
