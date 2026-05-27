import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import yaml
import argparse
import pickle

from utils.gpu_manager import is_main, barrier, broadcast_value


class DirFileManager:
    """A class for managing file operations and directory handling."""

    @staticmethod
    def save_config(save_path: Union[str, Path], args: argparse.Namespace):
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        with open(f"{save_path}/config.yaml", "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False)

    @staticmethod
    def open_npy(path: Union[str, Path]) -> np.ndarray:
        """Load a NumPy array from a .npy file."""
        return np.load(path, allow_pickle=True).item()

    @staticmethod
    def open_tokenizer(path: Union[str, Path]) -> Tuple[dict, dict]:
        """Open a pickled tokenizer file and return the vocabulary and merges."""
        with open(path, "rb") as f:
            vocab, merges = pickle.load(f)
        return vocab, merges

    @staticmethod
    def next_run_id(base: Union[str, Path]) -> str:
        base = Path(base)
        base.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
        return str(max(nums) + 1 if nums else 0)

    @staticmethod
    def ensure_directory_exists(
        folder: Optional[Union[str, Path]] = None,
        file: Optional[Union[str, Path]] = None,
    ) -> bool:
        """If `folder` is provided, ensure it exists and return True.
        If `file` is provided, ensure its parent dir exists and return whether the file exists.
        Exactly one of `folder` or `file` must be provided.
        """
        if (folder is None) == (file is None):
            raise ValueError("Provide exactly one of 'folder' or 'file'.")

        if folder is not None:
            d = Path(folder)
            d.mkdir(parents=True, exist_ok=True)
            return True

        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.exists()


def setup_experiment_folders(base_run_dir: Union[str, Path], args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Rank 0 picks run_id and creates both dirs, broadcasts run_id, then barrier.
    Everyone returns the same (config_dir, run_dir) as Paths.
    """
    base_run_dir = Path(base_run_dir)
    fm = DirFileManager()
    fm.ensure_directory_exists(folder=base_run_dir)

    if is_main():
        run_id = fm.next_run_id(base_run_dir)
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        fm.save_config(run_dir, args)
    else:
        run_id, run_dir = None, None

    run_id = broadcast_value(run_id, src=0)
    if not is_main():
        run_dir = base_run_dir / run_id

    barrier()
    return run_dir