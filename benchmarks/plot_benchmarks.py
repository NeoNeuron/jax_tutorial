"""
Visualise benchmark results from benchmarks/results/ and save figures to benchmarks/figures/.

Produces two figures:
  figures/benchmark_sweeps.pdf   — three-panel sweep: JAX-CPU, PyTorch-CPU, PyTorch-MPS
  figures/benchmark_speedup.pdf  — speedup ratios across all CPU configs (JAX vs PyTorch)

NOTE: JAX Metal (jax-metal) is incompatible with JAX 0.10.x, so GPU data is
PyTorch MPS only. The sweeps figure adds a third MPS line for direct comparison.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

JAX_CPU_COLOR   = '#2196F3'   # blue
TORCH_CPU_COLOR = '#F44336'   # red
TORCH_GPU_COLOR = '#FF9800'   # orange
ALPHA_ERR = 0.15


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results():
    records = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        data = json.loads(path.read_text())
        cfg = data['config']
        records.append({
            'framework': data['framework'],
            'model':     cfg['model'],
            'hidden':    cfg['hidden'],
            'seq_len':   cfg['seq_len'],
            'batch':     cfg['batch'],
            'rank':      cfg.get('rank'),
            'device':    cfg.get('device', 'cpu'),   # old files have no 'device' key → cpu
            'median_ms': data['median_ms'],
            'min_ms':    data['min_ms'],
            'max_ms':    data['max_ms'],
        })
    return records


def select(records, **filters):
    return [r for r in records if all(r.get(k) == v for k, v in filters.items())]


def _arrays(recs, x_key):
    recs = sorted(recs, key=lambda r: r[x_key])
    return (
        [r[x_key]        for r in recs],
        [r['median_ms']  for r in recs],
        [r['median_ms'] - r['min_ms'] for r in recs],
        [r['max_ms'] - r['median_ms'] for r in recs],
    )


def _plot_line(ax, recs, x_key, color, label, linestyle='-'):
    if not recs:
        return
    xs, med, lo, hi = _arrays(recs, x_key)
    ax.errorbar(xs, med, yerr=[lo, hi],
                color=color, marker='o', ms=5, lw=2,
                capsize=3, label=label, linestyle=linestyle)
    ax.fill_between(xs,
                     [m - l for m, l in zip(med, lo)],
                     [m + h for m, h in zip(med, hi)],
                     color=color, alpha=ALPHA_ERR)


# ---------------------------------------------------------------------------
# Figure 1 — three-panel sweep (CPU JAX, CPU Torch, GPU Torch)
# ---------------------------------------------------------------------------

def plot_sweeps(all_records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Training Step Time — JAX CPU vs PyTorch CPU vs PyTorch MPS (Apple M4 Pro, batch=64)',
                 fontsize=12)

    sweep_specs = [
        dict(ax=axes[0], x_key='seq_len', x_label='Sequence length', x_vals=[50, 100, 200, 500],
             title='A  Vanilla RNN — sequence length',
             fixed=dict(model='vanilla', hidden=64, batch=64)),
        dict(ax=axes[1], x_key='hidden',  x_label='Hidden size',     x_vals=[32, 64, 128, 256],
             title='B  Vanilla RNN — hidden size',
             fixed=dict(model='vanilla', seq_len=100, batch=64)),
        dict(ax=axes[2], x_key='rank',    x_label='Rank r',          x_vals=[1, 2, 4, 8, 16],
             title='C  Low-rank RNN — rank',
             fixed=dict(model='lowrank', hidden=64, seq_len=100, batch=64)),
    ]

    for spec in sweep_specs:
        ax    = spec['ax']
        fixed = spec['fixed']
        x_key = spec['x_key']

        _plot_line(ax, select(all_records, framework='jax',   device='cpu', **fixed), x_key,
                   JAX_CPU_COLOR,   'JAX (lax.scan + vmap, CPU)')
        _plot_line(ax, select(all_records, framework='torch', device='cpu', **fixed), x_key,
                   TORCH_CPU_COLOR, 'PyTorch (Python loop, CPU)')
        _plot_line(ax, select(all_records, framework='torch', device='gpu', **fixed), x_key,
                   TORCH_GPU_COLOR, 'PyTorch (Python loop, MPS)', linestyle='--')

        ax.set_title(spec['title'], fontsize=10, fontweight='bold')
        ax.set_xlabel(spec['x_label'])
        ax.set_ylabel('Median step time (ms)')
        ax.set_xticks(spec['x_vals'])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIGURES_DIR / f'benchmark_sweeps.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — speedup bar chart (CPU: JAX vs PyTorch)
# ---------------------------------------------------------------------------

def plot_speedup(all_records):
    def cfg_key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    cpu = [r for r in all_records if r['device'] == 'cpu']
    jax_map   = {cfg_key(r): r for r in cpu if r['framework'] == 'jax'}
    torch_map = {cfg_key(r): r for r in cpu if r['framework'] == 'torch'}
    shared    = sorted(set(jax_map) & set(torch_map))

    speedups = [torch_map[k]['median_ms'] / jax_map[k]['median_ms'] for k in shared]
    labels   = [_cfg_label(k) for k in shared]

    order    = np.argsort(speedups)[::-1]
    speedups = [speedups[i] for i in order]
    labels   = [labels[i]   for i in order]
    colors   = ['#2196F3' if 'lowrank' in labels[i] else '#4CAF50' for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.barh(range(len(labels)), speedups, color=colors, edgecolor='white', height=0.65)
    ax.axvline(1.0, color='grey', lw=1, ls='--')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('JAX speedup over PyTorch CPU  (PyTorch ms / JAX ms)', fontsize=10)
    ax.set_title('JAX CPU vs PyTorch CPU — Speedup (higher = JAX faster)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}×', va='center', fontsize=8.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2196F3', label='Low-rank RNN'),
        Patch(color='#4CAF50', label='Vanilla RNN'),
    ], fontsize=9, loc='lower right')
    ax.set_xlim(0, max(speedups) * 1.15)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIGURES_DIR / f'benchmark_speedup.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — CPU vs MPS comparison for PyTorch
# ---------------------------------------------------------------------------

def plot_cpu_vs_mps(all_records):
    """Bar chart: PyTorch CPU median vs MPS median for every config."""
    def cfg_key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    torch_cpu = {cfg_key(r): r for r in all_records if r['framework'] == 'torch' and r['device'] == 'cpu'}
    torch_gpu = {cfg_key(r): r for r in all_records if r['framework'] == 'torch' and r['device'] == 'gpu'}
    shared    = sorted(set(torch_cpu) & set(torch_gpu))

    ratios  = [torch_cpu[k]['median_ms'] / torch_gpu[k]['median_ms'] for k in shared]
    labels  = [_cfg_label(k) for k in shared]
    cpu_ms  = [torch_cpu[k]['median_ms'] for k in shared]
    mps_ms  = [torch_gpu[k]['median_ms'] for k in shared]

    # sort by CPU time descending for readability
    order  = np.argsort(cpu_ms)[::-1]
    labels = [labels[i] for i in order]
    ratios = [ratios[i] for i in order]
    cpu_ms = [cpu_ms[i] for i in order]
    mps_ms = [mps_ms[i] for i in order]

    y = np.arange(len(labels))
    bar_h = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PyTorch CPU vs MPS (Apple M4 Pro Metal GPU)', fontsize=12, fontweight='bold')

    # Left panel: absolute times
    ax = axes[0]
    ax.barh(y + bar_h/2, cpu_ms, height=bar_h, color=TORCH_CPU_COLOR, label='CPU')
    ax.barh(y - bar_h/2, mps_ms, height=bar_h, color=TORCH_GPU_COLOR, label='MPS')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Median step time (ms)')
    ax.set_title('Absolute times', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    # Right panel: CPU/MPS ratio (>1 = MPS faster)
    ax2 = axes[1]
    colors2 = [TORCH_CPU_COLOR if r < 1 else TORCH_GPU_COLOR for r in ratios]
    bars2 = ax2.barh(y, ratios, color=colors2, edgecolor='white', height=0.65)
    ax2.axvline(1.0, color='grey', lw=1, ls='--')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel('CPU time / MPS time  (> 1 means MPS is faster)', fontsize=10)
    ax2.set_title('CPU / MPS speedup ratio', fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, ratios):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}×', va='center', fontsize=8)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = FIGURES_DIR / f'benchmark_cpu_vs_mps.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved {out}')
    plt.close(fig)


def _cfg_label(k):
    model, hidden, seq_len, batch, rank = k
    base = f'{model}  h={hidden}  T={seq_len}'
    if rank is not None:
        base += f'  r={rank}'
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    records = load_results()
    print(f'Loaded {len(records)} benchmark records')
    by_device = {d: sum(1 for r in records if r['device'] == d) for d in ('cpu', 'gpu')}
    print(f'  CPU records: {by_device["cpu"]}  GPU records: {by_device["gpu"]}')
    plot_sweeps(records)
    plot_speedup(records)
    plot_cpu_vs_mps(records)
    print('Done.')
