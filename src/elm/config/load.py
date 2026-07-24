from pathlib import Path
from typing import Any
import argparse
import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str | Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()

    if path in stack:
        chain = " -> ".join(map(str, (*stack, path)))
        raise ValueError(f"Circular config dependency: {chain}")

    with path.open() as file:
        raw = yaml.safe_load(file) or {}

    config: dict[str, Any] = {}

    for default in raw.pop("defaults", []):
        included = load_config(path.parent / default, (*stack, path))
        config = deep_merge(config, included)

    return deep_merge(config, raw)

def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print("CONFIG:\n", config, "\n", "==="*30)
    return config