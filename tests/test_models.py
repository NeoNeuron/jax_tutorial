"""
Shape and behaviour tests for JAX RNN models (no PyTorch dependency).
Run with: pytest tests/test_models.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rnn_jax import (
    VanillaRNNCell, VanillaRNN, VanillaRNNModel,
    LowRankRNNCell, LowRankRNN, LowRankRNNModel,
)
from src.ode_solvers import solve_euler, solve_rk4, harmonic_oscillator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rngs():
    return nnx.Rngs(0)


# ---------------------------------------------------------------------------
# VanillaRNN
# ---------------------------------------------------------------------------

class TestVanillaRNNCell:

    def test_output_shape(self, rngs):
        cell = VanillaRNNCell(input_size=3, hidden_size=16, rngs=rngs)
        h = jnp.zeros((4, 16))   # (batch, hidden)
        x = jnp.ones((4, 3))
        h_new = cell(h, x)
        assert h_new.shape == (4, 16)

    def test_tanh_range(self, rngs):
        cell = VanillaRNNCell(input_size=3, hidden_size=16, rngs=rngs)
        h = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
        x = jax.random.normal(jax.random.PRNGKey(1), (32, 3))
        h_new = cell(h, x)
        assert float(jnp.max(jnp.abs(h_new))) <= 1.0 + 1e-6, 'tanh output must be in [-1, 1]'

    def test_deterministic(self, rngs):
        cell = VanillaRNNCell(input_size=2, hidden_size=8, rngs=rngs)
        h = jnp.zeros((1, 8))
        x = jnp.ones((1, 2))
        assert jnp.allclose(cell(h, x), cell(h, x))


class TestVanillaRNN:

    def test_output_shapes(self, rngs):
        B, T, I, H = 8, 20, 3, 16
        model = VanillaRNNModel(I, H, output_size=1, rngs=rngs)
        xs = jnp.ones((B, T, I))
        out = model(xs)
        assert out.shape == (B, T, 1)

    def test_scan_matches_loop(self, rngs):
        """vmap+scan output must match a nested Python loop over batch and time."""
        B, T, I, H = 4, 10, 2, 8
        rnn = VanillaRNN(I, H, rngs=rngs)
        xs = jax.random.normal(jax.random.PRNGKey(0), (B, T, I))

        # vmap+scan (the model's forward pass)
        outputs_scan, _ = rnn(xs)   # (B, T, H)

        # Reference: nested Python loops over batch then time
        graphdef, state = nnx.split(rnn.cell)
        outputs_loop = []
        for b in range(B):
            h = jnp.zeros(H)
            seq_out = []
            for t in range(T):
                cell = nnx.merge(graphdef, state)
                h = cell(h, xs[b, t])   # single example, single step
                seq_out.append(h)
            outputs_loop.append(jnp.stack(seq_out))
        outputs_loop = jnp.stack(outputs_loop)  # (B, T, H)

        assert jnp.allclose(outputs_scan, outputs_loop, atol=1e-5)

    def test_gradients_flow(self, rngs):
        model = VanillaRNNModel(1, 16, 1, rngs=rngs)
        xs = jax.random.normal(jax.random.PRNGKey(0), (4, 10, 1))  # (B, T, I)

        def loss(model):
            return jnp.mean(model(xs) ** 2)

        _, grads = nnx.value_and_grad(loss)(model)
        grad_w = grads.rnn.cell.linear.kernel[...]
        assert not jnp.all(grad_w == 0), 'Gradients should not all be zero'
        assert jnp.all(jnp.isfinite(grad_w)), 'Gradients must be finite'


# ---------------------------------------------------------------------------
# LowRankRNN
# ---------------------------------------------------------------------------

class TestLowRankRNNCell:

    def test_output_shape(self, rngs):
        cell = LowRankRNNCell(input_size=3, hidden_size=16, rank=4, rngs=rngs)
        h = jnp.zeros((4, 16))
        x = jnp.ones((4, 3))
        assert cell(h, x).shape == (4, 16)

    def test_rank_constraint(self, rngs):
        N, r = 32, 4
        cell = LowRankRNNCell(1, N, r, rngs=rngs)
        W_rec = np.array(cell.W_rec)
        svs = np.linalg.svd(W_rec, compute_uv=False)
        numerical_rank = int(np.sum(svs > 1e-6))
        assert numerical_rank == r, f'Expected rank {r}, got {numerical_rank}'

    @pytest.mark.parametrize('rank', [1, 2, 4, 8])
    def test_various_ranks(self, rank, rngs):
        cell = LowRankRNNCell(2, 32, rank=rank, rngs=nnx.Rngs(rank))
        W = np.array(cell.W_rec)
        svs = np.linalg.svd(W, compute_uv=False)
        assert int(np.sum(svs > 1e-6)) == rank

    def test_gradients_flow(self, rngs):
        model = LowRankRNNModel(1, 16, rank=4, output_size=1, rngs=rngs)
        xs = jax.random.normal(jax.random.PRNGKey(0), (4, 10, 1))  # (B, T, I)

        def loss(model):
            return jnp.mean(model(xs) ** 2)

        _, grads = nnx.value_and_grad(loss)(model)
        grad_U = grads.rnn.cell.U[...]
        grad_V = grads.rnn.cell.V[...]
        assert jnp.all(jnp.isfinite(grad_U))
        assert jnp.all(jnp.isfinite(grad_V))


class TestLowRankRNN:

    def test_scan_matches_loop(self, rngs):
        B, T, I, H, R = 4, 8, 2, 16, 3
        rnn = LowRankRNN(I, H, R, rngs=rngs)
        xs = jax.random.normal(jax.random.PRNGKey(0), (B, T, I))

        outputs_scan, _ = rnn(xs)   # (B, T, H)

        graphdef, state = nnx.split(rnn.cell)
        outputs_loop = []
        for b in range(B):
            h = jnp.zeros(H)
            seq_out = []
            for t in range(T):
                cell = nnx.merge(graphdef, state)
                h = cell(h, xs[b, t])
                seq_out.append(h)
            outputs_loop.append(jnp.stack(seq_out))
        outputs_loop = jnp.stack(outputs_loop)  # (B, T, H)

        assert jnp.allclose(outputs_scan, outputs_loop, atol=1e-5)


# ---------------------------------------------------------------------------
# ODE Solvers
# ---------------------------------------------------------------------------

class TestODESolvers:

    def test_euler_harmonic_shape(self):
        t = jnp.linspace(0, 5, 100)
        y0 = jnp.array([1.0, 0.0])
        ys = solve_euler(harmonic_oscillator, y0, t)
        assert ys.shape == (99, 2)

    def test_rk4_harmonic_shape(self):
        t = jnp.linspace(0, 5, 100)
        y0 = jnp.array([1.0, 0.0])
        ys = solve_rk4(harmonic_oscillator, y0, t)
        assert ys.shape == (99, 2)

    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be significantly more accurate than Euler."""
        import functools
        omega = 1.0
        t = jnp.linspace(0, 10, 200)
        y0 = jnp.array([1.0, 0.0])
        f = functools.partial(harmonic_oscillator, omega=omega)

        ys_euler = solve_euler(f, y0, t)
        ys_rk4   = solve_rk4(f, y0, t)
        exact_x  = jnp.cos(omega * t[1:])

        err_euler = float(jnp.mean((ys_euler[:, 0] - exact_x) ** 2))
        err_rk4   = float(jnp.mean((ys_rk4[:, 0]  - exact_x) ** 2))
        assert err_rk4 < err_euler / 10, f'RK4 ({err_rk4:.2e}) should be 10x more accurate than Euler ({err_euler:.2e})'

    def test_rk4_differentiable(self):
        """Gradients should flow through solve_rk4."""
        import functools
        t = jnp.linspace(0, 2, 50)
        y0 = jnp.array([1.0, 0.0])

        def loss(omega_scalar):
            f = functools.partial(harmonic_oscillator, omega=omega_scalar)
            ys = solve_rk4(f, y0, t)
            return jnp.mean(ys ** 2)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad)
