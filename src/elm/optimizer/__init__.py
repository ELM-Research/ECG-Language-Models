from importlib import import_module

def build_optimizer(config, model):
    module = import_module(f".{config['optimizer']['name']}", __name__)
    return module.build(config, model)
