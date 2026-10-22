"""Pure PyTorch input-selective bidirectional state-space block."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    """A compact Mamba-style diagonal SSM with input-dependent B, C and step."""

    def __init__(self, d_model: int, d_inner: int = 256, d_state: int = 16) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner)
        self.select_proj = nn.Linear(d_model, 1 + 2 * d_state)
        self.a_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.d = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        nn.init.uniform_(self.a_log, -3.0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, C], got {tuple(x.shape)}")
        normalized = self.norm(x)
        u = F.silu(self.in_proj(normalized))
        selector = self.select_proj(normalized)
        dt = F.softplus(selector[..., :1] - 3.0)
        b_t, c_t = selector[..., 1:].chunk(2, dim=-1)
        a = -torch.exp(self.a_log.float()).to(dtype=x.dtype)
        state = torch.zeros(
            x.shape[0],
            self.d_inner,
            self.d_state,
            dtype=x.dtype,
            device=x.device,
        )
        outputs: list[Tensor] = []
        # Sequence length is the small number of body rows, so an explicit scan
        # is both transparent and practical without a custom CUDA kernel.
        for step in range(x.shape[1]):
            delta = dt[:, step].unsqueeze(-1)
            decay = torch.exp(delta * a.unsqueeze(0))
            drive = (
                delta
                * b_t[:, step].unsqueeze(1)
                * u[:, step].unsqueeze(-1)
            )
            state = decay * state + drive
            y = (state * c_t[:, step].unsqueeze(1)).sum(dim=-1)
            outputs.append(y + self.d * u[:, step])
        return self.out_proj(torch.stack(outputs, dim=1))


class BidirectionalSelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_inner: int = 256, d_state: int = 16) -> None:
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_inner, d_state)
        self.backward_ssm = SelectiveSSM(d_model, d_inner, d_state)
        self.mix = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        forward = self.forward_ssm(x)
        backward = self.backward_ssm(x.flip(1)).flip(1)
        return self.mix(torch.cat((forward, backward), dim=-1))
