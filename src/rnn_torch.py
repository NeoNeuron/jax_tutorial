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
    """Single leaky RNN step: h_t = (1-α) h_{t-1} + α (W tanh(h_{t-1}) + W_in x_t + b)."""

    def __init__(self, input_size: int, hidden_size: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        std = 1.0 / (hidden_size ** 0.5)
        self.W    = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * std)
        self.b    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pre = torch.tanh(h) @ self.W.T + x @ self.W_in.T + self.b
        return (1.0 - self.alpha) * h + self.alpha * pre


class VanillaRNN(nn.Module):
    """
    Vanilla RNN rolled over a full sequence (Python loop).

    Args:
        xs: (B, T, input_size)
    Returns:
        outputs: (B, T, hidden_size)
        h_final: (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.cell = VanillaRNNCell(input_size, hidden_size, alpha=alpha)
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

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, alpha: float = 0.2
    ) -> None:
        super().__init__()
        self.rnn = VanillaRNN(input_size, hidden_size, alpha=alpha)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(xs)          # (B, T, hidden)
        return self.readout(hidden)        # (B, T, output_size)


# ---------------------------------------------------------------------------
# Low-Rank RNN
# ---------------------------------------------------------------------------

class LowRankRNNCell(nn.Module):
    """
    Low-rank RNN cell: J = M @ N.T stored as separate M, N matrices.

    h_t = (1-α) h_{t-1} + α (tanh(h_{t-1}) @ N @ M.T + x @ W_in.T + b)
    """

    def __init__(self, input_size: int, hidden_size: int, rank: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha

        std = 1.0 / (hidden_size ** 0.5)
        self.M    = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.N    = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * std)
        self.b    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pre = torch.tanh(h) @ self.N @ self.M.T + x @ self.W_in.T + self.b
        return (1.0 - self.alpha) * h + self.alpha * pre

    @property
    def J_rec(self) -> torch.Tensor:
        return self.M @ self.N.T


class LowRankRNN(nn.Module):
    """
    Low-rank RNN over a full sequence (Python loop).

    Args:
        xs: (B, T, input_size)
    Returns:
        outputs: (B, T, hidden_size)
        h_final: (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, rank: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.cell = LowRankRNNCell(input_size, hidden_size, rank, alpha=alpha)
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
        self, input_size: int, hidden_size: int, rank: int, output_size: int, alpha: float = 0.2
    ) -> None:
        super().__init__()
        self.rnn = LowRankRNN(input_size, hidden_size, rank, alpha=alpha)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(xs)          # (B, T, hidden)
        return self.readout(hidden)        # (B, T, output_size)


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

def load_vanilla_rnn_weights(
    torch_model: VanillaRNN,
    W: np.ndarray,
    W_in: np.ndarray,
    b: np.ndarray,
) -> None:
    """Load JAX VanillaRNNCell weights into a PyTorch VanillaRNNCell.

    W: (hidden, hidden), W_in: (hidden, input), b: (hidden,) — all same shapes in both frameworks.
    """
    with torch.no_grad():
        torch_model.cell.W.copy_(torch.tensor(W))
        torch_model.cell.W_in.copy_(torch.tensor(W_in))
        torch_model.cell.b.copy_(torch.tensor(b))


def load_lowrank_rnn_weights(
    torch_model: LowRankRNN,
    M: np.ndarray,
    N: np.ndarray,
    W_in: np.ndarray,
    b: np.ndarray,
) -> None:
    """Load JAX LowRankRNNCell weights into a PyTorch LowRankRNNCell.

    M: (hidden, rank), N: (hidden, rank), W_in: (hidden, input), b: (hidden,).
    """
    with torch.no_grad():
        torch_model.cell.M.copy_(torch.tensor(M))
        torch_model.cell.N.copy_(torch.tensor(N))
        torch_model.cell.W_in.copy_(torch.tensor(W_in))
        torch_model.cell.b.copy_(torch.tensor(b))
