"""Shared utilities: data generation, weight conversion, plotting."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic sequence tasks
# ---------------------------------------------------------------------------

def make_sinusoid_task(
    n_sequences: int,
    seq_len: int,
    freq_range: tuple[float, float] = (0.05, 0.3),
    key: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """
    Generate a batch of sinusoid prediction tasks.

    Input:  x_t = sin(2π f t)
    Target: x_{t+1}  (one-step-ahead prediction)

    Returns dict with keys 'inputs' (B, T, 1) and 'targets' (B, T, 1).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    freqs = jax.random.uniform(key, (n_sequences,), minval=freq_range[0], maxval=freq_range[1])
    t = jnp.arange(seq_len + 1)
    # shape: (B, T+1)
    signals = jnp.sin(2 * jnp.pi * freqs[:, None] * t[None, :])
    # inputs: (B, T, 1), targets: (B, T, 1)
    inputs  = signals[:, :-1, None]
    targets = signals[:, 1:,  None]
    return {"inputs": inputs, "targets": targets}


def make_copy_task(
    n_sequences: int,
    seq_len: int,
    n_classes: int = 8,
    delay: int = 10,
    key: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """
    Copy task: network reads a sequence, waits `delay` steps, then outputs the sequence.
    Inputs/targets are one-hot encoded.

    Returns dict with 'inputs' (T_total, B, n_classes+1) and 'targets' (T_total, B, n_classes+1).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    T_total = seq_len + delay + seq_len
    seq = jax.random.randint(key, (n_sequences, seq_len), 0, n_classes)

    def build(s):
        blank = jnp.full((delay,), n_classes, dtype=jnp.int32)
        zeros = jnp.zeros((seq_len,), dtype=jnp.int32)
        inp = jnp.concatenate([s, blank, zeros])
        tgt = jnp.concatenate([zeros, blank, s])
        return jax.nn.one_hot(inp, n_classes + 1), jax.nn.one_hot(tgt, n_classes + 1)

    inputs, targets = jax.vmap(build)(seq)  # (B, T, C)
    return {"inputs": inputs, "targets": targets}


# ---------------------------------------------------------------------------
# Weight conversion: JAX (flax.nnx) ↔ PyTorch
# ---------------------------------------------------------------------------

def jax_to_numpy(x: jax.Array) -> np.ndarray:
    return np.array(x)


def numpy_to_jax(x: np.ndarray) -> jax.Array:
    return jnp.array(x)


def torch_linear_to_jax(weight: Any, bias: Any | None) -> dict[str, np.ndarray]:
    """
    Convert a torch.nn.Linear layer's weight/bias to a JAX-compatible dict.
    PyTorch Linear stores weight as (out, in); JAX/Flax Linear as (in, out).
    """
    import torch
    w = weight.detach().cpu().numpy().T  # (in, out)
    result: dict[str, np.ndarray] = {"kernel": w}
    if bias is not None:
        result["bias"] = bias.detach().cpu().numpy()
    return result


def jax_kernel_to_torch(kernel: jax.Array):
    """Convert a JAX Linear kernel (in, out) to PyTorch weight (out, in)."""
    import torch
    return torch.tensor(np.array(kernel).T)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(
    losses: list[float],
    title: str = "Training loss",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def plot_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_examples: int = 3,
    title: str = "Predictions vs targets",
) -> plt.Figure:
    """Plot first n_examples sequences. targets/predictions shape: (B, T, 1)."""
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples), sharex=True)
    if n_examples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(targets[i, :, 0], label="target", lw=2)
        ax.plot(predictions[i, :, 0], "--", label="predicted", lw=2)
        ax.legend(fontsize=8)
        ax.set_title(f"Sequence {i}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_eigenspectrum(W: np.ndarray, title: str = "Eigenspectrum of W_rec") -> plt.Figure:
    """Plot eigenvalues of a square matrix in the complex plane."""
    eigvals = np.linalg.eigvals(W)
    fig, ax = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="gray")
    ax.add_patch(circle)
    ax.scatter(eigvals.real, eigvals.imag, s=30, zorder=3)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig
