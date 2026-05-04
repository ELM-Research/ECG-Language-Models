"""FSDP-aware Muon: handles DTensor (sharded) parameters.

Newton-Schulz orthogonalization is not shardable, so for each DTensor
parameter we all-gather to the full matrix, run NS on every rank (duplicated
compute, deterministic), then write the local slice back. Non-DTensor
parameters fall through to upstream Muon's functional path unchanged.

Momentum buffers stay sharded (DTensor) to match the parameter — momentum
updates are pointwise and so are local-shard correct.
"""

import torch
from torch.optim import Muon
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz, muon as _muon_fn


class MuonDistributed(Muon):

    @torch.no_grad()
    def step(self, closure=None):
        from torch.distributed.tensor import DTensor

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            local_params, local_grads, local_bufs = [], [], []
            dtensor_params = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("Muon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )

                if isinstance(p, DTensor) or isinstance(p.grad, DTensor):
                    dtensor_params.append(p)
                else:
                    local_params.append(p)
                    local_grads.append(p.grad)
                    local_bufs.append(state["momentum_buffer"])

            if local_params:
                _muon_fn(
                    local_params, local_grads, local_bufs,
                    lr=group["lr"], weight_decay=group["weight_decay"],
                    momentum=group["momentum"], nesterov=group["nesterov"],
                    ns_coefficients=group["ns_coefficients"],
                    eps=group["eps"], ns_steps=group["ns_steps"],
                    adjust_lr_fn=group["adjust_lr_fn"], has_complex=False,
                )

            for p in dtensor_params:
                self._dtensor_step(p, group)

        return loss

    @torch.no_grad()
    def _dtensor_step(self, p, group):
        from torch.distributed.tensor import DTensor, distribute_tensor

        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        state = self.state[p]
        buf = state["momentum_buffer"]
        grad = p.grad

        # Match upstream: buf <- mom*buf + (1-mom)*grad (pointwise, shard-local).
        buf.lerp_(grad, 1 - mom)
        update = grad.lerp(buf, mom) if group["nesterov"] else buf

        update_full = update.full_tensor() if isinstance(update, DTensor) else update
        update_full = _zeropower_via_newtonschulz(
            update_full, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], p.shape)

        p_full = p.full_tensor() if isinstance(p, DTensor) else p
        p_full = p_full * (1 - lr * wd) - adjusted_lr * update_full

        if isinstance(p, DTensor):
            new_dt = distribute_tensor(p_full, p.device_mesh, p.placements)
            p.to_local().copy_(new_dt.to_local())
        else:
            p.copy_(p_full)
