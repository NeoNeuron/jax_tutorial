"""
CLI benchmark script: JAX vs PyTorch training step timing.

Usage:
    python benchmarks/run_benchmarks.py --framework both --model vanilla --hidden 64 --seq-len 100 --batch 64
    python benchmarks/run_benchmarks.py --framework both --model lowrank --hidden 64 --rank 4 --seq-len 100 --batch 64
    python benchmarks/run_benchmarks.py --device gpu --framework torch --model vanilla --hidden 64 --seq-len 100 --batch 64

Devices:
    cpu  — JAX CPU backend, PyTorch CPU tensors (default)
    gpu  — PyTorch MPS (Apple Silicon Metal) only.
           JAX Metal (jax-metal) requires JAX 0.4.x and is incompatible with
           the JAX 0.10.x used in this project; JAX GPU runs are skipped.

Output: JSON file in benchmarks/results/ named
        {framework}_{model}_{device}_{config_hash}.json
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def bench_jax(model_type, hidden, seq_len, batch, rank, device='cpu', n_warmup=5, n_steps=20):
    if device == 'gpu':
        print('  [SKIP] JAX Metal requires jax-metal which is incompatible with JAX 0.10.x.')
        return None

    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import optax

    from src.rnn_jax import VanillaRNNModel, LowRankRNNModel
    from src.train import mse_loss, make_train_step

    if model_type == 'vanilla':
        model = VanillaRNNModel(1, hidden, 1, rngs=nnx.Rngs(0))
    else:
        model = LowRankRNNModel(1, hidden, rank, 1, rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    def loss_fn(model, batch):
        return mse_loss(model(batch['inputs']), batch['targets'])

    train_step = make_train_step(loss_fn)

    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (batch, seq_len, 1))
    data = {'inputs': xs, 'targets': xs}

    # Warmup (includes JIT compilation)
    t_compile_start = time.perf_counter()
    for _ in range(n_warmup):
        train_step(model, optimizer, data)
    jax.effects_barrier()
    compile_ms = (time.perf_counter() - t_compile_start) * 1000

    # Timed steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        train_step(model, optimizer, data)
        jax.effects_barrier()
        step_times.append((time.perf_counter() - t0) * 1000)

    return {
        'warmup_total_ms': compile_ms,
        'median_ms': statistics.median(step_times),
        'mean_ms': statistics.mean(step_times),
        'min_ms': min(step_times),
        'max_ms': max(step_times),
        'step_times_ms': step_times,
    }


def bench_torch(model_type, hidden, seq_len, batch, rank, device='cpu', n_warmup=5, n_steps=20):
    import torch
    from src.rnn_torch import VanillaRNNModel, LowRankRNNModel

    if device == 'gpu':
        if not torch.backends.mps.is_available():
            print('  [SKIP] PyTorch MPS not available on this machine.')
            return None
        torch_device = torch.device('mps')
    else:
        torch_device = torch.device('cpu')

    if model_type == 'vanilla':
        model = VanillaRNNModel(1, hidden, 1).to(torch_device)
    else:
        model = LowRankRNNModel(1, hidden, rank, 1).to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(batch, seq_len, 1, device=torch_device)

    def _sync():
        if device == 'gpu':
            torch.mps.synchronize()

    def step():
        optimizer.zero_grad()
        out = model(xs)
        loss = (out - xs).pow(2).mean()
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    # Warmup
    t_warmup = time.perf_counter()
    for _ in range(n_warmup):
        step()
    _sync()
    warmup_ms = (time.perf_counter() - t_warmup) * 1000

    # Timed steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
        _sync()
        step_times.append((time.perf_counter() - t0) * 1000)

    return {
        'warmup_total_ms': warmup_ms,
        'median_ms': statistics.median(step_times),
        'mean_ms': statistics.mean(step_times),
        'min_ms': min(step_times),
        'max_ms': max(step_times),
        'step_times_ms': step_times,
    }


def main():
    parser = argparse.ArgumentParser(description='JAX vs PyTorch RNN benchmark')
    parser.add_argument('--framework', choices=['jax', 'torch', 'both'], default='both')
    parser.add_argument('--model', choices=['vanilla', 'lowrank'], default='vanilla')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        help='cpu: CPU (both frameworks); gpu: MPS/Metal (PyTorch only — see docstring)')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4, help='Only used for lowrank model')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()

    config = {
        'model':   args.model,
        'hidden':  args.hidden,
        'seq_len': args.seq_len,
        'batch':   args.batch,
        'rank':    args.rank if args.model == 'lowrank' else None,
        'device':  args.device,
    }
    config_str  = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frameworks = ['jax', 'torch'] if args.framework == 'both' else [args.framework]

    for fw in frameworks:
        print(f'\n[{fw.upper()}] device={args.device}  model={args.model}  hidden={args.hidden}  '
              f'seq_len={args.seq_len}  batch={args.batch}'
              + (f'  rank={args.rank}' if args.model == 'lowrank' else ''))

        if fw == 'jax':
            stats = bench_jax(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                               args.device, args.n_warmup, args.n_steps)
        else:
            stats = bench_torch(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                                args.device, args.n_warmup, args.n_steps)

        if stats is None:
            continue

        print(f'  warmup total: {stats["warmup_total_ms"]:.0f} ms  '
              f'median step: {stats["median_ms"]:.2f} ms  '
              f'(min {stats["min_ms"]:.2f} / max {stats["max_ms"]:.2f})')

        record = {'framework': fw, 'config': config, **stats}
        out_path = args.out_dir / f'{fw}_{args.model}_{args.device}_{config_hash}.json'
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f'  Saved → {out_path}')


if __name__ == '__main__':
    main()
