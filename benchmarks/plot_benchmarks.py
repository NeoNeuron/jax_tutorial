"""
Visualise benchmark results from benchmarks/results/ and save figures to benchmarks/figures/.

Produces two files:
  figures/benchmark_sweeps.pdf   — three-panel sweep comparison (JAX vs PyTorch)
  figures/benchmark_speedup.pdf  — speedup ratios (JAX / PyTorch) across all configs
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

JAX_COLOR   = '#2196F3'   # blue
TORCH_COLOR = '#F44336'   # red
ALPHA_ERR   = 0.18


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results():
    records = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        data = json.loads(path.read_text())
        records.append({
            'framework': data['framework'],
            'model':     data['config']['model'],
            'hidden':    data['config']['hidden'],
            'seq_len':   data['config']['seq_len'],
            'batch':     data['config']['batch'],
            'rank':      data['config'].get('rank'),
            'median_ms': data['median_ms'],
            'min_ms':    data['min_ms'],
            'max_ms':    data['max_ms'],
        })
    return records


def select(records, **filters):
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def split_fw(records):
    jax   = {r['median_ms']: r for r in records if r['framework'] == 'jax'}
    torch = {r['median_ms']: r for r in records if r['framework'] == 'torch'}
    return (
        [r for r in records if r['framework'] == 'jax'],
        [r for r in records if r['framework'] == 'torch'],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arrays(recs, x_key):
    recs = sorted(recs, key=lambda r: r[x_key])
    xs      = [r[x_key]   for r in recs]
    medians = [r['median_ms'] for r in recs]
    lo      = [r['median_ms'] - r['min_ms']  for r in recs]
    hi      = [r['max_ms']  - r['median_ms'] for r in recs]
    return xs, medians, lo, hi


def sweep_panel(ax, all_records, x_key, x_label, title, x_vals, fixed):
    recs   = select(all_records, **fixed)
    jax_r, torch_r = split_fw(recs)

    for fw_recs, color, label in [
        (jax_r,   JAX_COLOR,   'JAX (lax.scan + vmap)'),
        (torch_r, TORCH_COLOR, 'PyTorch (Python loop)'),
    ]:
        xs, med, lo, hi = _arrays(fw_recs, x_key)
        ax.errorbar(xs, med, yerr=[lo, hi],
                    color=color, marker='o', ms=6, lw=2,
                    capsize=4, label=label)
        ax.fill_between(xs,
                         [m - l for m, l in zip(med, lo)],
                         [m + h for m, h in zip(med, hi)],
                         color=color, alpha=ALPHA_ERR)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Median step time (ms)')
    ax.set_xticks(x_vals)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Figure 1 — three-panel sweep
# ---------------------------------------------------------------------------

def plot_sweeps(all_records):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle('JAX vs PyTorch — Training Step Time (CPU, batch=64)', fontsize=13)

    # Panel A: seq_len sweep (vanilla, hidden=64)
    sweep_panel(axes[0], all_records,
                x_key='seq_len', x_label='Sequence length',
                title='A  Vanilla RNN — sequence length',
                x_vals=[50, 100, 200, 500],
                fixed=dict(model='vanilla', hidden=64, batch=64))

    # Panel B: hidden sweep (vanilla, seq_len=100)
    sweep_panel(axes[1], all_records,
                x_key='hidden', x_label='Hidden size',
                title='B  Vanilla RNN — hidden size',
                x_vals=[32, 64, 128, 256],
                fixed=dict(model='vanilla', seq_len=100, batch=64))

    # Panel C: rank sweep (lowrank, hidden=64, seq_len=100)
    sweep_panel(axes[2], all_records,
                x_key='rank', x_label='Rank r',
                title='C  Low-rank RNN — rank',
                x_vals=[1, 2, 4, 8, 16],
                fixed=dict(model='lowrank', hidden=64, seq_len=100, batch=64))

    plt.tight_layout()
    out = FIGURES_DIR / 'benchmark_sweeps.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    out_png = FIGURES_DIR / 'benchmark_sweeps.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved {out_png}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — speedup bar chart
# ---------------------------------------------------------------------------

def plot_speedup(all_records):
    # Pair JAX and PyTorch records that share (model, hidden, seq_len, batch, rank)
    def key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    jax_map   = {key(r): r for r in all_records if r['framework'] == 'jax'}
    torch_map = {key(r): r for r in all_records if r['framework'] == 'torch'}
    shared    = sorted(set(jax_map) & set(torch_map))

    speedups = [torch_map[k]['median_ms'] / jax_map[k]['median_ms'] for k in shared]
    labels   = [_config_label(k) for k in shared]

    # Sort by speedup descending
    order    = np.argsort(speedups)[::-1]
    speedups = [speedups[i] for i in order]
    labels   = [labels[i]   for i in order]

    # Color by model type
    colors = [JAX_COLOR if 'lowrank' in labels[i] else '#4CAF50'
              for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.barh(range(len(labels)), speedups, color=colors, edgecolor='white', height=0.65)
    ax.axvline(1.0, color='grey', lw=1, ls='--')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('JAX speedup over PyTorch  (PyTorch ms / JAX ms)', fontsize=10)
    ax.set_title('JAX vs PyTorch Speedup (higher = JAX faster)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}×', va='center', fontsize=8.5)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=JAX_COLOR,   label='Low-rank RNN'),
        Patch(color='#4CAF50',   label='Vanilla RNN'),
    ], fontsize=9, loc='lower right')

    ax.set_xlim(0, max(speedups) * 1.15)
    plt.tight_layout()
    out = FIGURES_DIR / 'benchmark_speedup.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')
    out_png = FIGURES_DIR / 'benchmark_speedup.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved {out_png}')
    plt.close(fig)


def _config_label(k):
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
    plot_sweeps(records)
    plot_speedup(records)
    print('Done.')
