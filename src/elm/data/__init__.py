from elm.data.build import DataBuilder


def build_data(config: dict):
    builder = DataBuilder(config)
    tokenizer = builder.build_llm_tokenizer()
    return tokenizer, builder.build_dataloader(tokenizer)
