"""
RNN models built with flax.nnx.

Two architectures:
  - VanillaRNN   : standard Elman RNN with tanh nonlinearity
  - LowRankRNN   : recurrent weight constrained to rank-r  (W_rec = U @ V.T)

Both use jax.lax.scan for sequential computation so they are JIT-compilable,
differentiable, and work with jax.vmap over batches.

NNX pattern for scan
────────────────────
flax.nnx modules hold mutable state, but lax.scan needs a pure function.
We extract the module's pytree state with nnx.split / nnx.merge outside the
scan, capture the (constant) params as a closure, and reconstruct the module
inside the step function. This is the idiomatic NNX approach.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx


# ---------------------------------------------------------------------------
# Vanilla RNN
# ---------------------------------------------------------------------------

class VanillaRNNCell(nnx.Module):
    """Single Elman RNN step: h_t = tanh(W [h_{t-1}; x_t] + b)."""

    def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs) -> None:
        self.hidden_size = hidden_size
        # fused linear: maps [h; x] -> hidden
        self.linear = nnx.Linear(hidden_size + input_size, hidden_size, rngs=rngs)

    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        """
        Args:
            h: previous hidden state, shape (..., hidden_size)
            x: input at current step, shape (..., input_size)
        Returns:
            h_new: updated hidden state, shape (..., hidden_size)
        """
        return jnp.tanh(self.linear(jnp.concatenate([h, x], axis=-1)))


class VanillaRNN(nnx.Module):
    """
    Vanilla RNN over a full sequence using lax.scan.

    Usage:
        model = VanillaRNN(input_size=1, hidden_size=64, rngs=nnx.Rngs(0))
        outputs, h_final = model(xs)  # xs: (B, T, input_size)
    """

    def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs) -> None:
        self.cell = VanillaRNNCell(input_size, hidden_size, rngs)
        self.hidden_size = hidden_size

    def __call__(
        self, xs: jax.Array, h0: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            xs: input sequence, shape (B, T, input_size)
            h0: initial hidden state (B, hidden_size); zeros if None

        Returns:
            outputs:  hidden states at each step, shape (B, T, hidden_size)
            h_final:  final hidden state, shape (B, hidden_size)
        """
        B, T, _ = xs.shape
        if h0 is None:
            h0 = jnp.zeros((B, self.hidden_size))

        # Extract params once; they are shared (constant) across the vmap batch.
        # Each vmap call processes one sequence; lax.scan steps through time.
        graphdef, state = nnx.split(self.cell)

        def run_single(x_seq: jax.Array, h_init: jax.Array):
            # x_seq: (T, input_size),  h_init: (hidden_size,)
            def step(h, x):
                cell = nnx.merge(graphdef, state)
                h_new = cell(h, x)
                return h_new, h_new
            h_final, outputs = jax.lax.scan(step, h_init, x_seq)
            return outputs, h_final   # (T, hidden_size), (hidden_size,)

        outputs, h_final = jax.vmap(run_single)(xs, h0)  # (B, T, H), (B, H)
        return outputs, h_final

    def init_hidden(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_size))


# ---------------------------------------------------------------------------
# Low-Rank RNN
# ---------------------------------------------------------------------------

class LowRankRNNCell(nnx.Module):
    """
    Low-rank RNN cell.

    The recurrent weight matrix is factored as W_rec = U @ V.T where
    U, V ∈ R^{hidden × rank}.  This constrains network dynamics to an
    r-dimensional subspace, which is interpretable and analytically tractable
    (useful in computational neuroscience and systems analysis).

    Forward pass:
        h_t = tanh( h_{t-1} @ V @ U.T  +  x_t @ W_in  +  b )
             = tanh( W_rec h_{t-1}      +  W_in x_t    +  b )
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_size = hidden_size
        self.rank = rank

        init = nnx.initializers.normal(stddev=1.0 / (hidden_size ** 0.5))
        self.U = nnx.Param(init(rngs.params(), (hidden_size, rank)))  # right factor
        self.V = nnx.Param(init(rngs.params(), (hidden_size, rank)))  # left factor
        self.W_in = nnx.Linear(input_size, hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        """
        Args:
            h: previous hidden state, shape (..., hidden_size)
            x: input, shape (..., input_size)
        Returns:
            h_new: shape (..., hidden_size)
        """
        # h @ V  → (..., rank);  then @ U.T → (..., hidden_size)
        recurrent = h @ self.V[...] @ self.U[...].T
        return jnp.tanh(recurrent + self.W_in(x))

    @property
    def W_rec(self) -> jax.Array:
        """Reconstruct the full recurrent weight matrix (hidden, hidden)."""
        return self.U[...] @ self.V[...].T


class LowRankRNN(nnx.Module):
    """
    Low-rank RNN over a full sequence using lax.scan.

    Usage:
        model = LowRankRNN(input_size=1, hidden_size=64, rank=4, rngs=nnx.Rngs(0))
        outputs, h_final = model(xs)   # xs: (B, T, input_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.cell = LowRankRNNCell(input_size, hidden_size, rank, rngs)
        self.hidden_size = hidden_size
        self.rank = rank

    def __call__(
        self, xs: jax.Array, h0: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            xs: input sequence, shape (B, T, input_size)
            h0: initial hidden state (B, hidden_size); zeros if None

        Returns:
            outputs:  shape (B, T, hidden_size)
            h_final:  shape (B, hidden_size)
        """
        B, T, _ = xs.shape
        if h0 is None:
            h0 = jnp.zeros((B, self.hidden_size))

        graphdef, state = nnx.split(self.cell)

        def run_single(x_seq: jax.Array, h_init: jax.Array):
            def step(h, x):
                cell = nnx.merge(graphdef, state)
                h_new = cell(h, x)
                return h_new, h_new
            h_final, outputs = jax.lax.scan(step, h_init, x_seq)
            return outputs, h_final

        outputs, h_final = jax.vmap(run_single)(xs, h0)  # (B, T, H), (B, H)
        return outputs, h_final

    def init_hidden(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_size))

    @property
    def W_rec(self) -> jax.Array:
        return self.cell.W_rec

    def effective_rank(self) -> int:
        """Numerical rank of W_rec (should equal self.rank by construction)."""
        import numpy as np
        return int(np.linalg.matrix_rank(np.array(self.W_rec)))


# ---------------------------------------------------------------------------
# Readout head (shared by both RNN types)
# ---------------------------------------------------------------------------

class LinearReadout(nnx.Module):
    """Simple linear projection from hidden state to output."""

    def __init__(self, hidden_size: int, output_size: int, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(hidden_size, output_size, rngs=rngs)

    def __call__(self, h: jax.Array) -> jax.Array:
        return self.linear(h)


# ---------------------------------------------------------------------------
# Full model wrappers (RNN + readout)
# ---------------------------------------------------------------------------

class VanillaRNNModel(nnx.Module):
    """VanillaRNN + linear readout, ready to train end-to-end."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.rnn = VanillaRNN(input_size, hidden_size, rngs)
        self.readout = LinearReadout(hidden_size, output_size, rngs)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """xs: (B, T, input_size) → predictions: (B, T, output_size)"""
        hidden, _ = self.rnn(xs)
        return self.readout(hidden)  # nnx.Linear broadcasts over (B, T) leading dims


class LowRankRNNModel(nnx.Module):
    """LowRankRNN + linear readout, ready to train end-to-end."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        output_size: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.rnn = LowRankRNN(input_size, hidden_size, rank, rngs)
        self.readout = LinearReadout(hidden_size, output_size, rngs)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """xs: (B, T, input_size) → predictions: (B, T, output_size)"""
        hidden, _ = self.rnn(xs)
        return self.readout(hidden)  # nnx.Linear broadcasts over (B, T) leading dims
