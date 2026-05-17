"""
Training utilities for flax.nnx models.

Provides:
  - mse_loss, cross_entropy_loss
  - make_train_step   : returns a JIT-compiled train_step closure for any nnx model
  - make_eval_step    : returns a JIT-compiled eval_step closure
  - train_epoch       : runs one epoch over a data dict
  - fit               : full training loop with loss history
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean((predictions - targets) ** 2)


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """
    logits: (..., num_classes)
    labels: (..., num_classes)  one-hot  OR  (...,) integer
    """
    if labels.ndim == logits.ndim:  # one-hot
        return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1))
    return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[..., labels])


# ---------------------------------------------------------------------------
# Generic train / eval steps
# ---------------------------------------------------------------------------

def make_train_step(
    loss_fn: Callable[[nnx.Module, dict], jax.Array],
) -> Callable:
    """
    Returns a JIT-compiled train_step(model, optimizer, batch) -> loss.

    The returned function mutates `model` and `optimizer` in-place (NNX style).
    """
    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_eval_step(
    loss_fn: Callable[[nnx.Module, dict], jax.Array],
) -> Callable:
    """Returns a JIT-compiled eval_step(model, batch) -> loss (no grad)."""
    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: dict[str, jax.Array],
    ) -> jax.Array:
        return loss_fn(model, batch)

    return eval_step


# ---------------------------------------------------------------------------
# Epoch / full fit loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_step_fn: Callable,
    batches: list[dict[str, jax.Array]],
) -> float:
    """Run one pass over all batches; return mean loss (as Python float)."""
    total = 0.0
    for batch in batches:
        loss = train_step_fn(model, optimizer, batch)
        total += float(loss)
    return total / len(batches)


def fit(
    model: nnx.Module,
    data: dict[str, jax.Array],
    *,
    loss_fn: Callable[[nnx.Module, dict], jax.Array],
    n_steps: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    log_every: int = 50,
    key: jax.Array | None = None,
) -> list[float]:
    """
    Train `model` for `n_steps` gradient steps.

    data must have at minimum keys 'inputs' and 'targets', each (T, N, *).
    Returns a list of per-step losses.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    train_step = make_train_step(loss_fn)

    N = data["inputs"].shape[0]  # dataset size (batch-first: axis 0 = sequences)
    losses: list[float] = []

    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, N)
        batch = {k: v[idx] for k, v in data.items()}
        loss = train_step(model, optimizer, batch)
        losses.append(float(loss))
        if (step + 1) % log_every == 0:
            print(f"  step {step+1:5d}/{n_steps}  loss = {losses[-1]:.6f}")

    return losses


# ---------------------------------------------------------------------------
# Gradient clipping helper
# ---------------------------------------------------------------------------

def make_optimizer(lr: float = 1e-3, max_grad_norm: float = 1.0) -> optax.GradientTransformation:
    """Adam with global gradient norm clipping."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr),
    )
