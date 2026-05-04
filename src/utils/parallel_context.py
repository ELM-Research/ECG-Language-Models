"""ParallelContext: 3D-aware (pp, dp, tp) device mesh and distributed state.

Today only the dp axis carries size > 1; tp/pp are always 1. The 3D mesh shape
is allocated up-front so that adding TP/PP later requires no API changes
elsewhere — call sites can reference ctx.dp_mesh / ctx.tp_mesh / ctx.pp_mesh
without knowing what's wired through them.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class ParallelContext:

    def __init__(self, strategy: str = "none"):
        if strategy not in {"none", "ddp", "fsdp"}:
            raise ValueError(f"unknown parallel strategy: {strategy}")
        self.strategy = strategy
        self._mesh: Optional[DeviceMesh] = None
        self._local_rank = 0
        self._init()

    def _init(self) -> None:
        if self.strategy == "none":
            return
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
        world = dist.get_world_size()
        self._mesh = init_device_mesh(
            "cuda", (1, world, 1), mesh_dim_names=("pp", "dp", "tp")
        )

    @property
    def is_distributed(self) -> bool:
        return self.strategy != "none"

    @property
    def world_size(self) -> int:
        if self._mesh is not None and dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def global_rank(self) -> int:
        if self._mesh is not None and dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def is_main(self) -> bool:
        return self.global_rank == 0

    @property
    def dp_mesh(self) -> Optional[DeviceMesh]:
        return self._mesh["dp"] if self._mesh is not None else None

    @property
    def tp_mesh(self) -> Optional[DeviceMesh]:
        return self._mesh["tp"] if self._mesh is not None else None

    @property
    def pp_mesh(self) -> Optional[DeviceMesh]:
        return self._mesh["pp"] if self._mesh is not None else None

    @property
    def dp_size(self) -> int:
        return self.dp_mesh.size() if self.dp_mesh is not None else 1

    @property
    def tp_size(self) -> int:
        return self.tp_mesh.size() if self.tp_mesh is not None else 1

    @property
    def pp_size(self) -> int:
        return self.pp_mesh.size() if self.pp_mesh is not None else 1

    @property
    def dp_rank(self) -> int:
        return self.dp_mesh.get_local_rank() if self.dp_mesh is not None else 0

    @property
    def tp_rank(self) -> int:
        return self.tp_mesh.get_local_rank() if self.tp_mesh is not None else 0

    @property
    def pp_rank(self) -> int:
        return self.pp_mesh.get_local_rank() if self.pp_mesh is not None else 0

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def cleanup(self) -> None:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except OSError:
                pass

    def broadcast_value(self, val, src: int = 0):
        if not (dist.is_available() and dist.is_initialized()):
            return val
        obj = [val]
        dist.broadcast_object_list(obj, src=src)
        return obj[0]


_CTX: Optional[ParallelContext] = None


def init_parallel_context(strategy: str = "none") -> ParallelContext:
    global _CTX
    if _CTX is None:
        _CTX = ParallelContext(strategy=strategy)
    return _CTX


def get_parallel_context() -> ParallelContext:
    global _CTX
    if _CTX is None:
        _CTX = ParallelContext(strategy="none")
    return _CTX


def reset_parallel_context() -> None:
    global _CTX
    _CTX = None
