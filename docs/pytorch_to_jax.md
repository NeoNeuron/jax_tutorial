# PyTorch → JAX/Flax Migration Guide

**Audience**: ML practitioners comfortable with PyTorch who want to learn JAX and Flax NNX.  
**Example**: Training a Vanilla RNN — the same model implemented in both frameworks side by side.

---

## Mental Model Shift

Before diving into code, three conceptual differences matter most:

| PyTorch | JAX / Flax NNX |
|---|---|
| Eager execution by default | Lazy / traced execution; `@nnx.jit` compiles once |
| Mutable tensors; in-place ops common | Immutable arrays; functions return new arrays |
| `autograd` attached to tensors | Functional transforms: `jax.grad`, `nnx.value_and_grad` |
| `model.zero_grad()` + `loss.backward()` + `optimizer.step()` | One call: `value_and_grad` → `optimizer.update` |
| `torch.manual_seed(n)` | Explicit PRNG keys threaded through every call |
| Python loop unrolls computation | `lax.scan` compiles the loop into a single XLA op |

---

## Block 1 — Module Definition

**PyTorch**: subclass `nn.Module`, define parameters in `__init__`, computation in `forward`.

```python
# PyTorch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, alpha: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        std = 1.0 / (hidden_size ** 0.5)
        self.W    = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * std)
        self.b    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h, x):
        pre = torch.tanh(h) @ self.W.T + x @ self.W_in.T + self.b
        return (1.0 - self.alpha) * h + self.alpha * pre
```

**JAX / Flax NNX**: subclass `nnx.Module`. Parameters are `nnx.Param` attributes. An `nnx.Rngs` handle threads PRNG keys explicitly — no global seed state.

```python
# JAX / Flax NNX
import jax.numpy as jnp
import flax.nnx as nnx

class VanillaRNNCell(nnx.Module):
    def __init__(self, input_size: int, hidden_size: int, rngs: nnx.Rngs, alpha: float = 0.2):
        self.hidden_size = hidden_size
        self.alpha = alpha
        init = nnx.initializers.normal(stddev=1.0 / (hidden_size ** 0.5))
        self.W    = nnx.Param(init(rngs.params(), (hidden_size, hidden_size)))
        self.W_in = nnx.Param(init(rngs.params(), (hidden_size, input_size)))
        self.b    = nnx.Param(jnp.zeros((hidden_size,)))

    def __call__(self, h, x):
        pre = jnp.tanh(h) @ self.W.T + x @ self.W_in.T + self.b
        return (1.0 - self.alpha) * h + self.alpha * pre
```

**Key differences**:
- No `super().__init__()` in NNX — not needed.
- No `nn.Parameter(...)` wrapping a tensor literal; instead `nnx.Param(array)` where the array comes from an explicit initializer called with a PRNG key.
- `forward` → `__call__`.
- `self.W.T` works identically — `.T` transposes in both frameworks.

---

## Block 2 — Sequential Computation (Loop vs `lax.scan`)

This is the most important architectural difference. PyTorch runs a Python `for` loop, dispatching one GPU kernel per step. JAX's `lax.scan` compiles the entire time loop into a single XLA program.

**PyTorch**: explicit Python loop, mutable hidden state.

```python
# PyTorch — VanillaRNN.forward
def forward(self, xs, h0=None):
    B, T, _ = xs.shape
    h = torch.zeros(B, self.hidden_size, device=xs.device) if h0 is None else h0
    outputs = []
    for t in range(T):           # T kernel dispatches on GPU
        h = self.cell(h, xs[:, t])
        outputs.append(h)
    return torch.stack(outputs, dim=1), h   # (B, T, H), (B, H)
```

**JAX**: `vmap` parallelises over the batch; `lax.scan` steps through time inside each sequence. The entire forward+backward pass for all T steps is one compiled kernel.

```python
# JAX — VanillaRNN.__call__
def __call__(self, xs, h0=None):
    B, T, _ = xs.shape
    if h0 is None:
        h0 = jnp.zeros((B, self.hidden_size))

    # Extract params once — they are constant across vmap and scan.
    graphdef, state = nnx.split(self.cell)

    def run_single(x_seq, h_init):     # processes ONE sequence (T, input_size)
        def step(h, x):
            cell = nnx.merge(graphdef, state)
            h_new = cell(h, x)
            return h_new, h_new         # (new carry, stacked output)
        h_final, outputs = jax.lax.scan(step, h_init, x_seq)
        return outputs, h_final         # (T, H), (H,)

    outputs, h_final = jax.vmap(run_single)(xs, h0)    # (B, T, H), (B, H)
    return outputs, h_final
```

**Why `nnx.split` / `nnx.merge`?**  
`lax.scan` requires a *pure function* (no mutable state). `nnx.split` extracts the module's parameter pytree as an immutable snapshot; `nnx.merge` reconstructs a fresh module inside the scan body. The parameters are captured in `state` as a closure — JAX sees them as constants and does not re-allocate them each step.

---

## Block 3 — Model Instantiation & PRNG

**PyTorch**: global seed, no key argument.

```python
# PyTorch
torch.manual_seed(42)
model = VanillaRNNModel(input_size=1, hidden_size=64, output_size=1)
```

**JAX**: pass an `nnx.Rngs` object constructed from an explicit `jax.random.PRNGKey`. Each call to `rngs.params()` consumes one derived key, making randomness fully reproducible and traceable.

```python
# JAX
import jax
model = VanillaRNNModel(
    input_size=1, hidden_size=64, output_size=1,
    rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
)
```

---

## Block 4 — Training Step

**PyTorch**: three-step ritual — zero gradients, backward, step.

```python
# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(model, optimizer, xs, targets):
    optimizer.zero_grad()
    preds = model(xs)
    loss = ((preds - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**JAX / Flax NNX**: one call to `nnx.value_and_grad` computes loss and gradients simultaneously. `optimizer.update` applies the gradient update in-place (NNX tracks mutable state internally).

```python
# JAX / Flax NNX
import optax

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

def loss_fn(model, batch):
    preds = model(batch["inputs"])
    return jnp.mean((preds - batch["targets"]) ** 2)

@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)   # mutates model in-place
    return loss
```

**Key differences**:
- No `zero_grad()` — JAX computes a fresh gradient pytree each time; there is no accumulated gradient to clear.
- `nnx.value_and_grad` returns `(loss, grads)` in one call. `grads` mirrors the model's pytree structure.
- `wrt=nnx.Param` tells the optimizer to only update learnable parameters (not, e.g., batch-norm stats stored as `nnx.Variable`).
- `@nnx.jit` compiles the entire train step (forward + backward + update) into one XLA program on first call.

---

## Block 5 — JIT Compilation

**PyTorch**: optional, via `torch.compile` (introduced in 2.0).

```python
# PyTorch
model = torch.compile(model)
```

**JAX**: `@nnx.jit` (or `@jax.jit`) decorates the function. Tracing happens on the *first* call with a given input shape; subsequent calls reuse the compiled binary.

```python
# JAX
@nnx.jit
def train_step(model, optimizer, batch):
    ...
```

JAX's JIT traces through Python control flow — avoid data-dependent Python `if`/`for` inside JIT-compiled functions. Use `jax.lax.cond` or `lax.scan` instead.

---

## Block 6 — Optimizer with Gradient Clipping

**PyTorch**: clip before `optimizer.step()`.

```python
# PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**JAX / Optax**: compose transforms with `optax.chain`. Clipping is part of the optimizer definition, not a separate call.

```python
# JAX / Optax
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# train_step is unchanged — clipping is applied inside optimizer.update
```

---

## Full Training Loop Comparison

**PyTorch**

```python
model = VanillaRNNModel(input_size=1, hidden_size=64, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(500):
    xs, targets = get_batch()          # (B, T, 1), (B, T, 1)
    optimizer.zero_grad()
    loss = ((model(xs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"step {step+1}  loss={loss.item():.4f}")
```

**JAX / Flax NNX**

```python
model = VanillaRNNModel(input_size=1, hidden_size=64, output_size=1,
                         rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

def loss_fn(model, batch):
    return jnp.mean((model(batch["inputs"]) - batch["targets"]) ** 2)

@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss

key = jax.random.PRNGKey(42)
for step in range(500):
    key, subkey = jax.random.split(key)
    batch = get_batch(subkey)           # pass key explicitly
    loss = train_step(model, optimizer, batch)
    if (step + 1) % 50 == 0:
        print(f"step {step+1}  loss={float(loss):.4f}")
```

---

## Quick Reference Card

| Task | PyTorch | JAX / Flax NNX |
|---|---|---|
| Define module | `class M(nn.Module)` | `class M(nnx.Module)` |
| Declare parameter | `nn.Parameter(tensor)` | `nnx.Param(array)` |
| Forward method | `def forward(self, x)` | `def __call__(self, x)` |
| Random init | `torch.manual_seed(n)` | `nnx.Rngs(params=PRNGKey(n))` |
| Sequence loop | `for t in range(T)` | `jax.lax.scan(step, h0, xs)` |
| Batch parallelism | Built into ops | `jax.vmap(fn)(batch)` |
| Compute gradient | `loss.backward()` | `nnx.value_and_grad(fn)(model, batch)` |
| Apply gradient | `optimizer.step()` | `optimizer.update(model, grads)` |
| Clear gradient | `optimizer.zero_grad()` | Not needed |
| JIT compile | `torch.compile(model)` | `@nnx.jit` |
| Gradient clipping | `clip_grad_norm_(...)` | `optax.clip_by_global_norm(...)` in chain |
| Device placement | `.to(device)` | Automatic; set via `jax.default_device` |
