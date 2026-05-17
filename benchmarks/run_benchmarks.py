"""
CLI benchmark script: JAX vs PyTorch training step timing.

Usage:
    python benchmarks/run_benchmarks.py --framework both --model vanilla --hidden 64 --seq-len 100 --batch 64
    python benchmarks/run_benchmarks.py --framework both --model lowrank --hidden 64 --rank 4 --seq-len 100 --batch 64

Output: JSON file in benchmarks/results/ with config + timing stats.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def bench_jax(model_type, hidden, seq_len, batch, rank, n_warmup=5, n_steps=20):
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
    xs = jax.random.normal(key, (batch, seq_len, 1))   # (B, T, I) batch-first
    data = {'inputs': xs, 'targets': xs}

    # Warmup (includes JIT compilation on first call)
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


def bench_torch(model_type, hidden, seq_len, batch, rank, n_warmup=5, n_steps=20):
    import torch
    from src.rnn_torch import VanillaRNNModel, LowRankRNNModel

    if model_type == 'vanilla':
        model = VanillaRNNModel(1, hidden, 1)
    else:
        model = LowRankRNNModel(1, hidden, rank, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(batch, seq_len, 1)   # (B, T, I) batch-first

    def step():
        optimizer.zero_grad()
        out = model(xs)
        loss = (out - xs).pow(2).mean()
        loss.backward()
        optimizer.step()
        return float(loss)

    # Warmup
    t_warmup = time.perf_counter()
    for _ in range(n_warmup):
        step()
    warmup_ms = (time.perf_counter() - t_warmup) * 1000

    # Timed steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
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
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4, help='Only used for lowrank model')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()

    config = {
        'model': args.model,
        'hidden': args.hidden,
        'seq_len': args.seq_len,
        'batch': args.batch,
        'rank': args.rank if args.model == 'lowrank' else None,
    }
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frameworks = ['jax', 'torch'] if args.framework == 'both' else [args.framework]

    for fw in frameworks:
        print(f'\n[{fw.upper()}] model={args.model}  hidden={args.hidden}  '
              f'seq_len={args.seq_len}  batch={args.batch}'
              + (f'  rank={args.rank}' if args.model == 'lowrank' else ''))

        if fw == 'jax':
            stats = bench_jax(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                               args.n_warmup, args.n_steps)
        else:
            stats = bench_torch(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                                args.n_warmup, args.n_steps)

        print(f'  warmup total: {stats["warmup_total_ms"]:.0f} ms  '
              f'median step: {stats["median_ms"]:.2f} ms  '
              f'(min {stats["min_ms"]:.2f} / max {stats["max_ms"]:.2f})')

        record = {'framework': fw, 'config': config, **stats}
        out_path = args.out_dir / f'{fw}_{args.model}_{config_hash}.json'
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f'  Saved → {out_path}')


if __name__ == '__main__':
    main()
