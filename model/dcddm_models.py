from __future__ import annotations

import torch
import torch.nn as nn

from model.tsmodels import get_network


class Dualmodel(nn.Module):
    """Dual-domain DCDDM classifier with temporal and frequency encoders."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.t_model = get_network(args)
        self.f_model = get_network(args)
        self.final_rep = self.t_model.final_rep
        self.mlp = nn.Linear(self.final_rep * 2, args.num_classes)

    def forward(self, x):
        x_f = torch.fft.rfft(x, dim=-1)
        x_f = torch.view_as_real(x_f).reshape(x_f.shape[0], x_f.shape[1], -1)
        x_f = x_f[:, :, : x.shape[-1]]
        t_emb = self.t_model.embed(x)
        f_emb = self.f_model.embed(x_f)
        out = self.mlp(torch.cat([t_emb, f_emb], dim=-1))
        return out, t_emb, f_emb

