from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.func import functional_call


class ReparamModule(nn.Module):
    """Small torchreparam-style wrapper for differentiable flat-param unrolls."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.param_infos = [(name, p.shape, p.numel()) for name, p in module.named_parameters()]
        self.buffer_dict = OrderedDict((name, b.detach().clone()) for name, b in module.named_buffers())

    def flat_param(self):
        return torch.cat([p.reshape(-1) for p in self.module.parameters()], 0)

    def unflatten(self, flat_param):
        offset = 0
        params = OrderedDict()
        for name, shape, numel in self.param_infos:
            params[name] = flat_param[offset : offset + numel].view(shape)
            offset += numel
        return params

    def forward(self, x, flat_param=None):
        if flat_param is None:
            return self.module(x)
        params_and_buffers = OrderedDict()
        params_and_buffers.update(self.unflatten(flat_param))
        params_and_buffers.update({k: v.to(x.device) for k, v in self.buffer_dict.items()})
        return functional_call(self.module, params_and_buffers, (x,), tie_weights=False)


def flatten_snapshot(snapshot, device):
    return torch.cat([p.data.reshape(-1).to(device) for p in snapshot], 0)
