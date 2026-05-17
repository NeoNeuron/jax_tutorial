"""
CLI benchmark script: JAX vs PyTorch training step timing.

Usage:
    # CPU (default)
    python benchmarks/run_benchmarks.py --framework both --model vanilla --hidden 64 --seq-len 100 --batch 64

    # Apple Silicon GPU (PyTorch MPS only — JAX Metal requires JAX 0.4.x)
    python benchmarks/run_benchmarks.py --device gpu --framework torch --model vanilla --hidden 64 --seq-len 100 --batch 64

    # NVIDIA GPU (both JAX CUDA and PyTorch CUDA)
    python benchmarks/run_benchmarks.py --device cuda --framework both --model vanilla --hidden 64 --seq-len 100 --batch 64

Device notes:
    cpu   — JAX CPU backend, PyTorch CPU tensors.
    gpu   — PyTorch MPS (Apple Silicon Metal). JAX Metal is skipped because
            jax-metal 0.1.1 is incompatible with JAX 0.10.x.
    cuda  — JAX CUDA + PyTorch CUDA (NVIDIA GPU). Requires:
              pip install "jax[cuda12]"   # or cuda11 depending on driver
            JAX auto-selects CUDA when installed; the script also sets
            JAX_PLATFORM_NAME=cuda to prevent fall-back to CPU.
            Peak GPU memory (MB) is recorded for both frameworks.

Output: JSON file in benchmarks/results/ named
        {framework}_{model}_{device}_{config_hash}.json
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# JAX platform selection — must happen before any `import jax`
# ---------------------------------------------------------------------------

def _configure_jax_platform(device: str) -> None:
    """Set JAX_PLATFORM_NAME before JAX initialises (lazy import in bench_jax)."""
    if device == 'cuda':
        os.environ['JAX_PLATFORM_NAME'] = 'cuda'
    elif device == 'cpu':
        # Force CPU even on a machine where jax-cuda is installed.
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    # 'gpu' (MPS): jax-metal unsupported, leave unset — JAX will stay on CPU.


# ---------------------------------------------------------------------------
# JAX benchmark
# ---------------------------------------------------------------------------

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

    if device == 'cuda':
        cuda_devices = jax.devices('cuda')
        if not cuda_devices:
            print('  [SKIP] No CUDA devices found by JAX. '
                  'Install jax[cuda12] and ensure NVIDIA drivers are available.')
            return None
        print(f'  JAX CUDA device: {cuda_devices[0]}')

    if model_type == 'vanilla':
        model = VanillaRNNModel(1, hidden, 1, rngs=nnx.Rngs(0))
    else:
        model = LowRankRNNModel(1, hidden, rank, 1, rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    def loss_fn(m, batch):
        return mse_loss(m(batch['inputs']), batch['targets'])

    train_step = make_train_step(loss_fn)

    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key, (batch, seq_len, 1))
    data = {'inputs': xs, 'targets': xs}

    # Warmup — includes JIT compilation; on CUDA also covers first-kernel overhead.
    t_compile_start = time.perf_counter()
    for _ in range(n_warmup):
        loss = train_step(model, optimizer, data)
        loss.block_until_ready()       # correct sync for both CPU and CUDA
    compile_ms = (time.perf_counter() - t_compile_start) * 1000

    # Timed steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        loss = train_step(model, optimizer, data)
        loss.block_until_ready()       # block until compute is done (not just dispatch)
        step_times.append((time.perf_counter() - t0) * 1000)

    stats = {
        'warmup_total_ms': compile_ms,
        'median_ms':       statistics.median(step_times),
        'mean_ms':         statistics.mean(step_times),
        'min_ms':          min(step_times),
        'max_ms':          max(step_times),
        'step_times_ms':   step_times,
    }

    # Peak GPU memory — JAX exposes this via device memory stats (CUDA only).
    if device == 'cuda':
        try:
            mem_stats = jax.devices('cuda')[0].memory_stats()
            # 'peak_bytes_in_use' is the high-water mark for the live allocator.
            peak_mb = mem_stats.get('peak_bytes_in_use', 0) / 1024 / 1024
            stats['peak_memory_mb'] = round(peak_mb, 2)
        except Exception:
            stats['peak_memory_mb'] = None

    return stats


# ---------------------------------------------------------------------------
# PyTorch benchmark
# ---------------------------------------------------------------------------

def bench_torch(model_type, hidden, seq_len, batch, rank, device='cpu', n_warmup=5, n_steps=20):
    import torch
    from src.rnn_torch import VanillaRNNModel, LowRankRNNModel

    if device == 'gpu':
        if not torch.backends.mps.is_available():
            print('  [SKIP] PyTorch MPS not available on this machine.')
            return None
        torch_device = torch.device('mps')
        def _sync(): torch.mps.synchronize()

    elif device == 'cuda':
        if not torch.cuda.is_available():
            print('  [SKIP] PyTorch CUDA not available. '
                  'Ensure an NVIDIA GPU is present and CUDA drivers are installed.')
            return None
        torch_device = torch.device('cuda')
        def _sync(): torch.cuda.synchronize()
        print(f'  PyTorch CUDA device: {torch.cuda.get_device_name(0)}')

    else:
        torch_device = torch.device('cpu')
        def _sync(): pass

    if model_type == 'vanilla':
        model = VanillaRNNModel(1, hidden, 1).to(torch_device)
    else:
        model = LowRankRNNModel(1, hidden, rank, 1).to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(batch, seq_len, 1, device=torch_device)

    def step():
        optimizer.zero_grad()
        out = model(xs)
        loss = (out - xs).pow(2).mean()
        loss.backward()
        optimizer.step()

    # Warmup
    t_warmup = time.perf_counter()
    for _ in range(n_warmup):
        step()
    _sync()
    warmup_ms = (time.perf_counter() - t_warmup) * 1000

    # Reset peak memory counter right before the timed run.
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Timed steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
        _sync()
        step_times.append((time.perf_counter() - t0) * 1000)

    stats = {
        'warmup_total_ms': warmup_ms,
        'median_ms':       statistics.median(step_times),
        'mean_ms':         statistics.mean(step_times),
        'min_ms':          min(step_times),
        'max_ms':          max(step_times),
        'step_times_ms':   step_times,
    }

    if device == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        stats['peak_memory_mb'] = round(peak_mb, 2)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='JAX vs PyTorch RNN benchmark')
    parser.add_argument('--framework', choices=['jax', 'torch', 'both'], default='both')
    parser.add_argument('--model', choices=['vanilla', 'lowrank'], default='vanilla')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'cuda'], default='cpu',
                        help='cpu: CPU; gpu: PyTorch MPS (Apple Silicon); cuda: JAX+PyTorch NVIDIA GPU')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4, help='Only used for lowrank model')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-steps', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()

    # Must happen before any `import jax` (bench_jax imports lazily).
    _configure_jax_platform(args.device)

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
        print(f'\n[{fw.upper()}] device={args.device}  model={args.model}  '
              f'hidden={args.hidden}  seq_len={args.seq_len}  batch={args.batch}'
              + (f'  rank={args.rank}' if args.model == 'lowrank' else ''))

        if fw == 'jax':
            stats = bench_jax(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                               args.device, args.n_warmup, args.n_steps)
        else:
            stats = bench_torch(args.model, args.hidden, args.seq_len, args.batch, args.rank,
                                args.device, args.n_warmup, args.n_steps)

        if stats is None:
            continue

        mem_str = (f'  peak GPU mem: {stats["peak_memory_mb"]:.1f} MB'
                   if stats.get('peak_memory_mb') is not None else '')
        print(f'  warmup total: {stats["warmup_total_ms"]:.0f} ms  '
              f'median step: {stats["median_ms"]:.2f} ms  '
              f'(min {stats["min_ms"]:.2f} / max {stats["max_ms"]:.2f}){mem_str}')

        record = {'framework': fw, 'config': config, **stats}
        out_path = args.out_dir / f'{fw}_{args.model}_{args.device}_{config_hash}.json'
        with open(out_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f'  Saved → {out_path}')


if __name__ == '__main__':
    main()
