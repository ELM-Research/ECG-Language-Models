"""Thin helpers around HuggingFace chat templates.

We rely entirely on each tokenizer's built-in chat template
(``tokenizer.apply_chat_template``), so supporting a new model needs no code
here: its tokenizer already ships the right template and special tokens. This
replaces the old (unmaintained) FastChat conversation templates and the
per-model hardcoding of BOS/EOS/response-start token ids.
"""


def _ids(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Token ids for a conversation via its chat template (return_dict=False -> a plain list)."""
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=add_generation_prompt)


def encode_with_labels(tokenizer, messages: list[dict], think_prefix: tuple[int, ...] = ()) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and label only assistant turns.

    Returns ``(input_ids, labels)`` where ``labels`` is ``input_ids`` on
    assistant turns (content + the template's turn terminator) and ``-100``
    everywhere else. Each assistant span is located purely from the template by
    diffing ``add_generation_prompt`` boundaries, so no per-model token
    bookkeeping is needed. ``think_prefix`` token ids, when present at the start
    of an assistant turn, are masked (explicit-thinking seeds them in the prompt).
    """
    ids = _ids(tokenizer, messages)
    labels = [-100] * len(ids)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        s = len(_ids(tokenizer, messages[:i], add_generation_prompt=True))
        e = len(_ids(tokenizer, messages[: i + 1]))
        labels[s:e] = ids[s:e]
        if think_prefix and ids[s : s + len(think_prefix)] == list(think_prefix):
            labels[s : s + len(think_prefix)] = [-100] * len(think_prefix)
    return ids, labels


def assistant_stop_ids(tokenizer) -> frozenset[int]:
    """Token ids that end an assistant turn for this chat template.

    Derived from the template (e.g. ``<|eot_id|>``, ``<|im_end|>``,
    ``<end_of_turn>``) plus the tokenizer EOS, so it works for any chat model
    without hardcoding. Used to stop and trim generation.
    """
    base = _ids(tokenizer, [{"role": "user", "content": "x"}], add_generation_prompt=True)
    full = _ids(tokenizer, [{"role": "user", "content": "x"}, {"role": "assistant", "content": "x"}])
    special = set(tokenizer.all_special_ids)
    stop = {t for t in full[len(base):] if t in special}
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    return frozenset(stop)
