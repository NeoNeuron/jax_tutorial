# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A layered JAX/Flax tutorial covering scientific computing (vmap, scan, ODE solving) and RNN training (Vanilla RNN, Low-rank RNN), with PyTorch numerical-consistency checks and benchmarks.

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GPU JAX (optional, replace cpu with cuda12 or tpu)
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run consistency tests with printed diffs
pytest tests/test_consistency.py -v -s

# Quick benchmark (writes JSON to benchmarks/results/)
python benchmarks/run_benchmarks.py --framework both --model vanilla --hidden 64 --seq-len 100 --batch 64

# Low-rank benchmark
python benchmarks/run_benchmarks.py --framework both --model lowrank --hidden 64 --rank 4 --seq-len 100 --batch 64

# Execute a notebook headlessly
jupyter nbconvert --to notebook --execute notebooks/01_jax_basics.ipynb --output notebooks/01_jax_basics_executed.ipynb
```

## Repository Layout

```
notebooks/          # 8 Jupyter notebooks (01–08), numbered by topic
src/
  rnn_jax.py        # VanillaRNN + LowRankRNN in flax.nnx
  rnn_torch.py      # Identical PyTorch implementations for comparison
  ode_solvers.py    # euler_step, rk4_step, solve_rk4, solve_rk4_batch
  train.py          # make_train_step, make_eval_step, fit loop
  utils.py          # data generation, weight conversion, plotting
benchmarks/
  run_benchmarks.py # CLI: sweeps config, writes JSON to benchmarks/results/
tests/
  test_models.py    # Shape checks, scan-vs-loop equivalence, ODE accuracy
  test_consistency.py  # JAX vs PyTorch output/gradient matching
```

## Key flax.nnx Patterns

**Module definition** — parameters are attributes, no `@nn.compact`:
```python
class MyCell(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
    def __call__(self, x):
        return jnp.tanh(self.linear(x))
```

**Data convention** — all sequence data is **batch-first**: `(B, T, I)`.
`vmap` parallelises over the batch dimension; `lax.scan` steps through time per sequence.

**vmap + lax.scan with NNX** — the standard pattern for batched RNNs:
```python
graphdef, state = nnx.split(self.cell)   # extract params once (shared across batch)

def run_single(x_seq, h_init):           # processes ONE sequence (T, I)
    def step(h, x):
        cell = nnx.merge(graphdef, state)
        h_new = cell(h, x)
        return h_new, h_new
    h_final, outputs = jax.lax.scan(step, h_init, x_seq)
    return outputs, h_final

outputs, h_final = jax.vmap(run_single)(xs, h0)  # xs: (B, T, I) → (B, T, H)
```

**Training** — NNX optimizer mutates model in-place:
```python
optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss
```

**Gradient extraction** — grads have the same pytree structure as the model:
```python
_, grads = nnx.value_and_grad(loss_fn)(model)
grad_kernel = grads.rnn.cell.linear.kernel[...]  # navigate like the model; use [...] not .value
```

## Low-Rank RNN

`W_rec = U @ V.T` where `U, V ∈ R^{N×r}`. Stored as `nnx.Param` on `LowRankRNNCell`.
The rank-r constraint holds by construction throughout training — no projection step needed.
Access the reconstructed matrix with `model.rnn.cell.W_rec` (property).

## JAX Timing Gotcha

JAX dispatches asynchronously. Always call `.block_until_ready()` or `jax.effects_barrier()` before stopping a timer, or you measure dispatch latency only.

## Weight Conversion (JAX ↔ PyTorch)

JAX `nnx.Linear` stores kernel as `(in, out)`. PyTorch `nn.Linear` stores weight as `(out, in)`.
When loading JAX weights into PyTorch: `torch_layer.weight.copy_(torch.tensor(jax_kernel.T))`.
Helper functions in `src/rnn_torch.py`: `load_vanilla_rnn_weights`, `load_lowrank_rnn_weights`.
