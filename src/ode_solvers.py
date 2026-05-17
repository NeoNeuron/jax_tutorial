"""
Numerical ODE solvers built with JAX.

All solvers use jax.lax.scan internally so they are JIT-compilable and
differentiable end-to-end. vmap over initial conditions is demonstrated in
notebook 03, but nothing here prevents it.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Step functions (pure, stateless)
# ---------------------------------------------------------------------------

def euler_step(
    f: Callable[[jax.Array, float], jax.Array],
    y: jax.Array,
    t: float,
    dt: float,
) -> jax.Array:
    """Single Euler step: y_{t+dt} = y_t + dt * f(y_t, t)."""
    return y + dt * f(y, t)


def rk4_step(
    f: Callable[[jax.Array, float], jax.Array],
    y: jax.Array,
    t: float,
    dt: float,
) -> jax.Array:
    """Single 4th-order Runge-Kutta step."""
    k1 = f(y,            t)
    k2 = f(y + dt/2 * k1, t + dt/2)
    k3 = f(y + dt/2 * k2, t + dt/2)
    k4 = f(y + dt   * k3, t + dt)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Full trajectory solvers (scan-based)
# ---------------------------------------------------------------------------

def solve_euler(
    f: Callable[[jax.Array, float], jax.Array],
    y0: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """
    Solve dy/dt = f(y, t) with Euler method over time array `t`.

    Args:
        f:  ODE right-hand side, signature f(y, t) -> dy/dt
        y0: initial state, shape (D,)
        t:  time points, shape (T,); assumed evenly spaced

    Returns:
        ys: trajectory, shape (T-1, D)  — states at t[1], t[2], ..., t[-1]
    """
    dt = t[1] - t[0]

    def step(y, t_curr):
        y_next = euler_step(f, y, t_curr, dt)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, t[:-1])
    return ys


def solve_rk4(
    f: Callable[[jax.Array, float], jax.Array],
    y0: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """
    Solve dy/dt = f(y, t) with RK4 over time array `t`.

    Args:
        f:  ODE right-hand side, signature f(y, t) -> dy/dt
        y0: initial state, shape (D,)
        t:  time points, shape (T,); assumed evenly spaced

    Returns:
        ys: trajectory, shape (T-1, D)  — states at t[1], ..., t[-1]
    """
    dt = t[1] - t[0]

    def step(y, t_curr):
        y_next = rk4_step(f, y, t_curr, dt)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, t[:-1])
    return ys


# ---------------------------------------------------------------------------
# Convenience: batch solving over many initial conditions
# ---------------------------------------------------------------------------

def solve_rk4_batch(
    f: Callable[[jax.Array, float], jax.Array],
    y0_batch: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """
    Solve the same ODE for a batch of initial conditions using vmap.

    Args:
        f:        ODE right-hand side (applies to a single trajectory)
        y0_batch: initial states, shape (B, D)
        t:        time points, shape (T,)

    Returns:
        trajectories: shape (B, T-1, D)
    """
    return jax.vmap(lambda y0: solve_rk4(f, y0, t))(y0_batch)


# ---------------------------------------------------------------------------
# Example ODEs for the tutorial notebooks
# ---------------------------------------------------------------------------

def harmonic_oscillator(y: jax.Array, t: float, omega: float = 1.0) -> jax.Array:
    """
    Simple harmonic oscillator: d²x/dt² = -ω² x.
    State y = [x, v].
    """
    x, v = y
    return jnp.array([v, -(omega**2) * x])


def lorenz(y: jax.Array, t: float, sigma: float = 10., rho: float = 28., beta: float = 8/3) -> jax.Array:
    """Lorenz system. State y = [x, y, z]."""
    x, yy, z = y
    dx = sigma * (yy - x)
    dy = x * (rho - z) - yy
    dz = x * yy - beta * z
    return jnp.array([dx, dy, dz])


def van_der_pol(y: jax.Array, t: float, mu: float = 1.0) -> jax.Array:
    """Van der Pol oscillator. State y = [x, v]."""
    x, v = y
    return jnp.array([v, mu * (1 - x**2) * v - x])
