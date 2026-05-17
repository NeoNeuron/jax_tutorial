"""
PyTorch mirror of the JAX RNN implementations.

Architectures and forward-pass semantics are kept deliberately identical so
that numerical consistency tests (tests/test_consistency.py) can load the
same weights into both frameworks and compare outputs/gradients.

Conventions
───────────
- Inputs:  (B, T, input_size)   (batch-first, matching the JAX convention)
- Outputs: (B, T, output_size)
- All models use tanh nonlinearity (matching the JAX versions)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Vanilla RNN
# ---------------------------------------------------------------------------

class VanillaRNNCell(nn.Module):
    """Single Elman step: h_t = tanh(W [h_{t-1}; x_t] + b)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(torch.cat([h, x], dim=-1)))


class VanillaRNN(nn.Module):
    """
    Vanilla RNN rolled over a full sequence (Python loop).

    Args:
        xs: (B, T, input_size)
    Returns:
        outputs: (B, T, hidden_size)
        h_final: (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = VanillaRNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        xs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = xs.shape
        h = torch.zeros(B, self.hidden_size, device=xs.device) if h0 is None else h0
        outputs = []
        for t in range(T):
            h = self.cell(h, xs[:, t])
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)   # (B, T, hidden_size)
        return outputs, h


class VanillaRNNModel(nn.Module):
    """VanillaRNN + linear readout."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = VanillaRNN(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(xs)          # (B, T, hidden)
        return self.readout(hidden)        # (B, T, output_size)


# ---------------------------------------------------------------------------
# Low-Rank RNN
# ---------------------------------------------------------------------------

class LowRankRNNCell(nn.Module):
    """
    Low-rank RNN cell: W_rec = U @ V.T stored as separate U, V matrices.

    h_t = tanh( h_{t-1} @ V @ U.T  +  x_t @ W_in.T  +  b )
    """

    def __init__(self, input_size: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank

        std = 1.0 / (hidden_size ** 0.5)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        recurrent = h @ self.V @ self.U.T
        return torch.tanh(recurrent + self.W_in(x))

    @property
    def W_rec(self) -> torch.Tensor:
        return self.U @ self.V.T


class LowRankRNN(nn.Module):
    """
    Low-rank RNN over a full sequence (Python loop).

    Args:
        xs: (B, T, input_size)
    Returns:
        outputs: (B, T, hidden_size)
        h_final: (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.cell = LowRankRNNCell(input_size, hidden_size, rank)
        self.hidden_size = hidden_size
        self.rank = rank

    def forward(
        self,
        xs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = xs.shape
        h = torch.zeros(B, self.hidden_size, device=xs.device) if h0 is None else h0
        outputs = []
        for t in range(T):
            h = self.cell(h, xs[:, t])
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)   # (B, T, hidden_size)
        return outputs, h


class LowRankRNNModel(nn.Module):
    """LowRankRNN + linear readout."""

    def __init__(
        self, input_size: int, hidden_size: int, rank: int, output_size: int
    ) -> None:
        super().__init__()
        self.rnn = LowRankRNN(input_size, hidden_size, rank)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(xs)          # (B, T, hidden)
        return self.readout(hidden)        # (B, T, output_size)


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

def load_vanilla_rnn_weights(
    torch_model: VanillaRNN,
    kernel: np.ndarray,
    bias: np.ndarray,
) -> None:
    """
    Load JAX-style weights into a PyTorch VanillaRNNCell.

    JAX kernel shape: (hidden + input, hidden)  →  PyTorch weight: (hidden, hidden+input)
    """
    with torch.no_grad():
        torch_model.cell.linear.weight.copy_(torch.tensor(kernel.T))
        torch_model.cell.linear.bias.copy_(torch.tensor(bias))


def load_lowrank_rnn_weights(
    torch_model: LowRankRNN,
    U: np.ndarray,
    V: np.ndarray,
    W_in_kernel: np.ndarray,
    W_in_bias: np.ndarray,
) -> None:
    """Load JAX LowRankRNNCell weights into a PyTorch LowRankRNNCell."""
    with torch.no_grad():
        torch_model.cell.U.copy_(torch.tensor(U))
        torch_model.cell.V.copy_(torch.tensor(V))
        torch_model.cell.W_in.weight.copy_(torch.tensor(W_in_kernel.T))
        torch_model.cell.W_in.bias.copy_(torch.tensor(W_in_bias))
