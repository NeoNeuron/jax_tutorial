"""
Numerical consistency tests: JAX (flax.nnx) vs PyTorch.

Loads identical weights into both frameworks, runs the same inputs,
and asserts that outputs and gradients match to float32 precision.

Run with: pytest tests/test_consistency.py -v -s
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rnn_jax import VanillaRNNModel, LowRankRNNModel
from src.rnn_torch import (
    VanillaRNNModel as TorchVanillaRNNModel,
    LowRankRNNModel as TorchLowRankRNNModel,
    load_vanilla_rnn_weights,
    load_lowrank_rnn_weights,
)

ATOL = 2e-5   # float32 tolerance


@pytest.fixture(scope='module')
def vanilla_models():
    """Return a JAX and PyTorch VanillaRNN with identical weights."""
    I, H = 3, 16
    jax_model = VanillaRNNModel(I, H, output_size=1, rngs=nnx.Rngs(0))

    W_np   = np.array(jax_model.rnn.cell.W)    # (H, H)
    Win_np = np.array(jax_model.rnn.cell.W_in) # (H, I)
    b_np   = np.array(jax_model.rnn.cell.b)    # (H,)
    ro_W   = np.array(jax_model.readout.linear.kernel)
    ro_b   = np.array(jax_model.readout.linear.bias)

    torch_model = TorchVanillaRNNModel(I, H, output_size=1)
    load_vanilla_rnn_weights(torch_model.rnn, W_np, Win_np, b_np)
    with torch.no_grad():
        torch_model.readout.weight.copy_(torch.tensor(ro_W.T))
        torch_model.readout.bias.copy_(torch.tensor(ro_b))

    return jax_model, torch_model, I, H


@pytest.fixture(scope='module')
def lowrank_models():
    """Return a JAX and PyTorch LowRankRNN with identical weights."""
    I, H, R = 3, 16, 4
    jax_model = LowRankRNNModel(I, H, rank=R, output_size=1, rngs=nnx.Rngs(0))

    M_np   = np.array(jax_model.rnn.cell.M)
    N_np   = np.array(jax_model.rnn.cell.N)
    Win_np = np.array(jax_model.rnn.cell.W_in)  # (H, I)
    b_np   = np.array(jax_model.rnn.cell.b)
    ro_W   = np.array(jax_model.readout.linear.kernel)
    ro_b   = np.array(jax_model.readout.linear.bias)

    torch_model = TorchLowRankRNNModel(I, H, rank=R, output_size=1)
    load_lowrank_rnn_weights(torch_model.rnn, M_np, N_np, Win_np, b_np)
    with torch.no_grad():
        torch_model.readout.weight.copy_(torch.tensor(ro_W.T))
        torch_model.readout.bias.copy_(torch.tensor(ro_b))

    return jax_model, torch_model, I, H, R


@pytest.fixture
def inputs():
    """Shared random inputs (B, T, I) — batch-first convention."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 15, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# VanillaRNN
# ---------------------------------------------------------------------------

class TestVanillaRNNConsistency:

    def test_forward_output(self, vanilla_models, inputs):
        jax_m, torch_m, _, _ = vanilla_models
        xs_jax   = jnp.array(inputs)
        xs_torch = torch.tensor(inputs)

        out_jax   = np.array(jax_m(xs_jax))
        out_torch = torch_m(xs_torch).detach().numpy()

        diff = np.max(np.abs(out_jax - out_torch))
        print(f'\n  Vanilla forward max diff: {diff:.2e}')
        assert diff < ATOL, f'Forward pass mismatch: max diff = {diff:.2e} > {ATOL}'

    def test_hidden_states(self, vanilla_models, inputs):
        jax_m, torch_m, _, H = vanilla_models
        xs_jax   = jnp.array(inputs)
        xs_torch = torch.tensor(inputs)

        jax_h, _   = jax_m.rnn(xs_jax)
        torch_h, _ = torch_m.rnn(xs_torch)

        jax_h_np   = np.array(jax_h)
        torch_h_np = torch_h.detach().numpy()

        diff = np.max(np.abs(jax_h_np - torch_h_np))
        print(f'\n  Vanilla hidden states max diff: {diff:.2e}')
        assert diff < ATOL

    def test_gradient_W(self, vanilla_models, inputs):
        jax_m, torch_m, _, _ = vanilla_models
        xs_jax   = jnp.array(inputs)
        xs_torch = torch.tensor(inputs)

        def jax_loss(m): return jnp.mean(m(xs_jax) ** 2)
        _, grads = nnx.value_and_grad(jax_loss)(jax_m)
        jax_gW = np.array(grads.rnn.cell.W)

        torch_m.zero_grad()
        torch_m(xs_torch).pow(2).mean().backward()
        torch_gW = torch_m.rnn.cell.W.grad.numpy()

        diff = np.max(np.abs(jax_gW - torch_gW))
        print(f'\n  Vanilla grad W max diff: {diff:.2e}')
        assert diff < ATOL

    def test_gradient_b(self, vanilla_models, inputs):
        jax_m, torch_m, _, _ = vanilla_models
        xs_jax   = jnp.array(inputs)
        xs_torch = torch.tensor(inputs)

        def jax_loss(m): return jnp.mean(m(xs_jax) ** 2)
        _, grads = nnx.value_and_grad(jax_loss)(jax_m)
        jax_gb = np.array(grads.rnn.cell.b)

        torch_m.zero_grad()
        torch_m(xs_torch).pow(2).mean().backward()
        torch_gb = torch_m.rnn.cell.b.grad.numpy()

        diff = np.max(np.abs(jax_gb - torch_gb))
        print(f'\n  Vanilla grad b max diff: {diff:.2e}')
        assert diff < ATOL


# ---------------------------------------------------------------------------
# LowRankRNN
# ---------------------------------------------------------------------------

class TestLowRankRNNConsistency:

    def test_forward_output(self, lowrank_models, inputs):
        jax_m, torch_m, I, H, R = lowrank_models
        xs_np = inputs[:, :, :I]
        xs_jax   = jnp.array(xs_np)
        xs_torch = torch.tensor(xs_np)

        out_jax   = np.array(jax_m(xs_jax))
        out_torch = torch_m(xs_torch).detach().numpy()

        diff = np.max(np.abs(out_jax - out_torch))
        print(f'\n  LowRank forward max diff: {diff:.2e}')
        assert diff < ATOL

    def test_gradient_M(self, lowrank_models, inputs):
        jax_m, torch_m, I, H, R = lowrank_models
        xs_np = inputs[:, :, :I]
        xs_jax   = jnp.array(xs_np)
        xs_torch = torch.tensor(xs_np)

        def jax_loss(m): return jnp.mean(m(xs_jax) ** 2)
        _, grads = nnx.value_and_grad(jax_loss)(jax_m)
        jax_gM = np.array(grads.rnn.cell.M)

        torch_m.zero_grad()
        torch_m(xs_torch).pow(2).mean().backward()
        torch_gM = torch_m.rnn.cell.M.grad.numpy()

        diff = np.max(np.abs(jax_gM - torch_gM))
        print(f'\n  LowRank grad M max diff: {diff:.2e}')
        assert diff < ATOL

    def test_gradient_N(self, lowrank_models, inputs):
        jax_m, torch_m, I, H, R = lowrank_models
        xs_np = inputs[:, :, :I]
        xs_jax   = jnp.array(xs_np)
        xs_torch = torch.tensor(xs_np)

        def jax_loss(m): return jnp.mean(m(xs_jax) ** 2)
        _, grads = nnx.value_and_grad(jax_loss)(jax_m)
        jax_gN = np.array(grads.rnn.cell.N)

        torch_m.zero_grad()
        torch_m(xs_torch).pow(2).mean().backward()
        torch_gN = torch_m.rnn.cell.N.grad.numpy()

        diff = np.max(np.abs(jax_gN - torch_gN))
        print(f'\n  LowRank grad N max diff: {diff:.2e}')
        assert diff < ATOL

    def test_gradient_Win(self, lowrank_models, inputs):
        jax_m, torch_m, I, H, R = lowrank_models
        xs_np = inputs[:, :, :I]
        xs_jax   = jnp.array(xs_np)
        xs_torch = torch.tensor(xs_np)

        def jax_loss(m): return jnp.mean(m(xs_jax) ** 2)
        _, grads = nnx.value_and_grad(jax_loss)(jax_m)
        jax_gWin = np.array(grads.rnn.cell.W_in)  # (H, I)

        torch_m.zero_grad()
        torch_m(xs_torch).pow(2).mean().backward()
        torch_gWin = torch_m.rnn.cell.W_in.grad.numpy()  # (H, I) — same shape

        diff = np.max(np.abs(jax_gWin - torch_gWin))
        print(f'\n  LowRank grad W_in max diff: {diff:.2e}')
        assert diff < ATOL
