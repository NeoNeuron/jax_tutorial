"""
Visualise benchmark results from benchmarks/results/ and save figures to benchmarks/figures/.

Produces four figures:
  benchmark_sweeps.pdf      — seq_len / hidden / rank sweeps, all device-framework combos
  benchmark_speedup.pdf     — JAX vs PyTorch speedup on CPU and CUDA
  benchmark_cpu_vs_mps.pdf  — PyTorch CPU vs MPS (Apple Silicon) comparison
  benchmark_cuda.pdf        — JAX CUDA vs PyTorch CUDA: sweeps, speedup, peak memory
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

# Colour / style palette
STYLE = {
    ('jax',   'cpu'):  dict(color='#2196F3', ls='-',  marker='o', label='JAX CPU  (lax.scan)'),
    ('jax',   'cuda'): dict(color='#0D47A1', ls='--', marker='s', label='JAX CUDA (lax.scan)'),
    ('torch', 'cpu'):  dict(color='#F44336', ls='-',  marker='o', label='PyTorch CPU  (loop)'),
    ('torch', 'gpu'):  dict(color='#FF9800', ls='--', marker='^', label='PyTorch MPS  (loop)'),
    ('torch', 'cuda'): dict(color='#9C27B0', ls='--', marker='D', label='PyTorch CUDA (loop)'),
}
ALPHA_BAND = 0.12


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results():
    records = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        data = json.loads(path.read_text())
        cfg = data['config']
        records.append({
            'framework':      data['framework'],
            'model':          cfg['model'],
            'hidden':         cfg['hidden'],
            'seq_len':        cfg['seq_len'],
            'batch':          cfg['batch'],
            'rank':           cfg.get('rank'),
            'device':         cfg.get('device', 'cpu'),
            'median_ms':      data['median_ms'],
            'min_ms':         data['min_ms'],
            'max_ms':         data['max_ms'],
            'peak_memory_mb': data.get('peak_memory_mb'),
        })
    return records


def select(records, **kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def _arrays(recs, x_key):
    recs = sorted(recs, key=lambda r: r[x_key])
    return (
        [r[x_key]        for r in recs],
        [r['median_ms']  for r in recs],
        [r['median_ms'] - r['min_ms']  for r in recs],
        [r['max_ms']    - r['median_ms'] for r in recs],
    )


def _plot_line(ax, recs, x_key, fw, device, ms=5):
    if not recs:
        return
    sty = STYLE[(fw, device)]
    xs, med, lo, hi = _arrays(recs, x_key)
    ax.errorbar(xs, med, yerr=[lo, hi],
                color=sty['color'], ls=sty['ls'], marker=sty['marker'],
                ms=ms, lw=2, capsize=3, label=sty['label'])
    ax.fill_between(xs,
                     [m - l for m, l in zip(med, lo)],
                     [m + h for m, h in zip(med, hi)],
                     color=sty['color'], alpha=ALPHA_BAND)


def _fmt_panel(ax, title, x_label, x_vals, x_key=None):
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel('Median step time (ms)', fontsize=9)
    if x_vals:
        ax.set_xticks(x_vals)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=1)


def _savefig(fig, stem):
    for ext in ('pdf', 'png'):
        out = FIGURES_DIR / f'{stem}.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — full sweep (all device × framework combos)
# ---------------------------------------------------------------------------

def plot_sweeps(all_records):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        'Training Step Time — JAX vs PyTorch across CPU / MPS / CUDA  (batch=64)',
        fontsize=12)

    sweep_specs = [
        dict(ax=axes[0], x_key='seq_len', x_label='Sequence length',
             x_vals=[50, 100, 200, 500],
             title='A  Vanilla RNN — sequence length',
             fixed=dict(model='vanilla', hidden=64, batch=64)),
        dict(ax=axes[1], x_key='hidden', x_label='Hidden size',
             x_vals=[32, 64, 128, 256],
             title='B  Vanilla RNN — hidden size',
             fixed=dict(model='vanilla', seq_len=100, batch=64)),
        dict(ax=axes[2], x_key='rank', x_label='Rank r',
             x_vals=[1, 2, 4, 8, 16],
             title='C  Low-rank RNN — rank',
             fixed=dict(model='lowrank', hidden=64, seq_len=100, batch=64)),
    ]

    combos = [('jax', 'cpu'), ('jax', 'cuda'),
              ('torch', 'cpu'), ('torch', 'gpu'), ('torch', 'cuda')]

    for spec in sweep_specs:
        ax = spec['ax']
        for fw, dev in combos:
            recs = select(all_records, framework=fw, device=dev, **spec['fixed'])
            _plot_line(ax, recs, spec['x_key'], fw, dev)
        _fmt_panel(ax, spec['title'], spec['x_label'], spec['x_vals'])

    plt.tight_layout()
    _savefig(fig, 'benchmark_sweeps')


# ---------------------------------------------------------------------------
# Figure 2 — speedup bars: JAX vs PyTorch on CPU and CUDA
# ---------------------------------------------------------------------------

def plot_speedup(all_records):
    def cfg_key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    def speedup_pairs(device):
        jax_map   = {cfg_key(r): r for r in all_records
                     if r['framework'] == 'jax'   and r['device'] == device}
        torch_map = {cfg_key(r): r for r in all_records
                     if r['framework'] == 'torch' and r['device'] == device}
        shared = sorted(set(jax_map) & set(torch_map))
        ratios = [torch_map[k]['median_ms'] / jax_map[k]['median_ms'] for k in shared]
        labels = [_cfg_label(k) for k in shared]
        return ratios, labels

    cpu_ratios,  cpu_labels  = speedup_pairs('cpu')
    cuda_ratios, cuda_labels = speedup_pairs('cuda')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('JAX Speedup over PyTorch  (PyTorch ms / JAX ms, same device)',
                 fontsize=12, fontweight='bold')

    for ax, ratios, labels, device, color, use_log in [
        (axes[0], cpu_ratios,  cpu_labels,  'CPU',  '#2196F3', True),
        (axes[1], cuda_ratios, cuda_labels, 'CUDA', '#0D47A1', False),
    ]:
        if not ratios:
            ax.set_visible(False)
            continue
        order   = np.argsort(ratios)[::-1]
        ratios  = [ratios[i]  for i in order]
        labels  = [labels[i]  for i in order]
        bar_colors = ['#2196F3' if 'lowrank' in labels[i] else '#4CAF50'
                      for i in range(len(labels))]
        bars = ax.barh(range(len(labels)), ratios,
                       color=bar_colors, edgecolor='white', height=0.65)
        ax.axvline(1.0, color='grey', lw=1, ls='--')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f'JAX {device} / PyTorch {device} speedup  (higher = JAX faster)',
                      fontsize=9)
        ax.set_title(f'{device}  —  JAX lax.scan vs PyTorch loop', fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        if use_log:
            ax.set_xscale('log')
            all_pos = [v for v in ratios if v > 0]
            lo = min(all_pos) * 0.5
            hi = max(all_pos) * 1.8
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            for bar, val in zip(bars, ratios):
                ax.text(val * 1.04, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}×', va='center', fontsize=8)
        else:
            ax.set_xlim(0, max(ratios) * 1.15)
            for bar, val in zip(bars, ratios):
                ax.text(bar.get_width() + max(ratios) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}×', va='center', fontsize=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#2196F3', label='Low-rank RNN'),
            Patch(color='#4CAF50', label='Vanilla RNN'),
        ], fontsize=8, loc='lower right')

    plt.tight_layout()
    _savefig(fig, 'benchmark_speedup')


# ---------------------------------------------------------------------------
# Figure 3 — PyTorch CPU vs MPS
# ---------------------------------------------------------------------------

def plot_cpu_vs_mps(all_records):
    def cfg_key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    cpu_map = {cfg_key(r): r for r in all_records
               if r['framework'] == 'torch' and r['device'] == 'cpu'}
    mps_map = {cfg_key(r): r for r in all_records
               if r['framework'] == 'torch' and r['device'] == 'gpu'}
    shared  = sorted(set(cpu_map) & set(mps_map))
    if not shared:
        print('No CPU+MPS pairs found — skipping benchmark_cpu_vs_mps')
        return

    cpu_ms = [cpu_map[k]['median_ms'] for k in shared]
    mps_ms = [mps_map[k]['median_ms'] for k in shared]
    ratios = [c / m for c, m in zip(cpu_ms, mps_ms)]
    labels = [_cfg_label(k) for k in shared]

    order  = np.argsort(cpu_ms)[::-1]
    labels = [labels[i] for i in order]
    ratios = [ratios[i] for i in order]
    cpu_ms = [cpu_ms[i] for i in order]
    mps_ms = [mps_ms[i] for i in order]

    y, bh = np.arange(len(labels)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PyTorch CPU vs MPS (Apple M4 Pro Metal GPU)', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.barh(y + bh/2, cpu_ms, height=bh, color='#F44336', label='CPU')
    ax.barh(y - bh/2, mps_ms, height=bh, color='#FF9800', label='MPS')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Median step time (ms)'); ax.set_title('Absolute times', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, axis='x', alpha=0.3)

    ax2 = axes[1]
    colors2 = ['#FF9800' if r > 1 else '#F44336' for r in ratios]
    bars2 = ax2.barh(y, ratios, color=colors2, edgecolor='white', height=0.65)
    ax2.axvline(1.0, color='grey', lw=1, ls='--')
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel('CPU / MPS  (> 1 means MPS is faster)', fontsize=10)
    ax2.set_title('CPU / MPS speedup ratio', fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, ratios):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}×', va='center', fontsize=8)

    plt.tight_layout()
    _savefig(fig, 'benchmark_cpu_vs_mps')


# ---------------------------------------------------------------------------
# Figure 4 — CUDA deep-dive: sweeps + speedup + peak memory
# ---------------------------------------------------------------------------

def plot_cuda(all_records):
    cuda = [r for r in all_records if r['device'] == 'cuda']
    if not cuda:
        print('No CUDA records found — skipping benchmark_cuda')
        return

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle('CUDA Benchmark — JAX (lax.scan) vs PyTorch (Python loop)  (batch=64)',
                 fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.38)

    # ── Top row: three sweep panels ──────────────────────────────────────
    sweep_specs = [
        dict(ax=fig.add_subplot(gs[0, 0]),
             x_key='seq_len', x_label='Sequence length', x_vals=[50, 100, 200, 500],
             title='A  Vanilla RNN — seq_len',
             fixed=dict(model='vanilla', hidden=64, batch=64)),
        dict(ax=fig.add_subplot(gs[0, 1]),
             x_key='hidden', x_label='Hidden size', x_vals=[32, 64, 128, 256],
             title='B  Vanilla RNN — hidden',
             fixed=dict(model='vanilla', seq_len=100, batch=64)),
        dict(ax=fig.add_subplot(gs[0, 2:]),
             x_key='rank', x_label='Rank r', x_vals=[1, 2, 4, 8, 16],
             title='C  Low-rank RNN — rank',
             fixed=dict(model='lowrank', hidden=64, seq_len=100, batch=64)),
    ]

    for spec in sweep_specs:
        ax = spec['ax']
        for fw, dev in [('jax', 'cuda'), ('torch', 'cuda')]:
            recs = select(cuda, framework=fw, device=dev, **spec['fixed'])
            _plot_line(ax, recs, spec['x_key'], fw, dev)
        _fmt_panel(ax, spec['title'], spec['x_label'], spec['x_vals'])

    # ── Bottom-left: speedup bars (JAX CUDA / PyTorch CUDA) ──────────────
    def cfg_key(r):
        return (r['model'], r['hidden'], r['seq_len'], r['batch'], r['rank'])

    jax_map   = {cfg_key(r): r for r in cuda if r['framework'] == 'jax'}
    torch_map = {cfg_key(r): r for r in cuda if r['framework'] == 'torch'}
    shared    = sorted(set(jax_map) & set(torch_map))

    speedups = [torch_map[k]['median_ms'] / jax_map[k]['median_ms'] for k in shared]
    s_labels = [_cfg_label(k) for k in shared]
    order    = np.argsort(speedups)[::-1]
    speedups = [speedups[i] for i in order]
    s_labels = [s_labels[i] for i in order]
    s_colors = ['#0D47A1' if 'lowrank' in s_labels[i] else '#1B5E20'
                for i in range(len(s_labels))]

    ax_spd = fig.add_subplot(gs[1, :2])
    bars = ax_spd.barh(range(len(s_labels)), speedups,
                       color=s_colors, edgecolor='white', height=0.65)
    ax_spd.axvline(1.0, color='grey', lw=1, ls='--')
    ax_spd.set_yticks(range(len(s_labels)))
    ax_spd.set_yticklabels(s_labels, fontsize=8)
    ax_spd.set_xlabel('JAX CUDA / PyTorch CUDA speedup  (higher = JAX faster)', fontsize=9)
    ax_spd.set_title('D  Speedup: JAX CUDA vs PyTorch CUDA', fontsize=10, fontweight='bold')
    ax_spd.grid(True, axis='x', alpha=0.3)
    ax_spd.set_xlim(0, max(speedups) * 1.14)
    for bar, val in zip(bars, speedups):
        ax_spd.text(bar.get_width() + max(speedups) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}×', va='center', fontsize=8)
    from matplotlib.patches import Patch
    ax_spd.legend(handles=[
        Patch(color='#0D47A1', label='Low-rank RNN'),
        Patch(color='#1B5E20', label='Vanilla RNN'),
    ], fontsize=8, loc='lower right')

    # ── Bottom-right: peak memory grouped bars ────────────────────────────
    mem_pairs = [(k, jax_map[k], torch_map[k]) for k in shared
                 if jax_map[k].get('peak_memory_mb') is not None
                 and torch_map[k].get('peak_memory_mb') is not None]
    # sort by JAX peak memory descending
    mem_pairs.sort(key=lambda t: t[1]['peak_memory_mb'], reverse=True)

    m_labels  = [_cfg_label(k)         for k, _, _ in mem_pairs]
    jax_mem   = [j['peak_memory_mb']   for _, j, _ in mem_pairs]
    torch_mem = [t['peak_memory_mb']   for _, _, t in mem_pairs]

    y_m = np.arange(len(m_labels))
    bh  = 0.35
    ax_mem = fig.add_subplot(gs[1, 2:])
    ax_mem.barh(y_m + bh/2, jax_mem,   height=bh, color='#0D47A1', label='JAX CUDA')
    ax_mem.barh(y_m - bh/2, torch_mem, height=bh, color='#9C27B0', label='PyTorch CUDA')
    ax_mem.set_yticks(y_m)
    ax_mem.set_yticklabels(m_labels, fontsize=8)
    ax_mem.set_xlabel('Peak GPU memory (MB)', fontsize=9)
    ax_mem.set_title('E  Peak GPU memory', fontsize=10, fontweight='bold')
    ax_mem.legend(fontsize=8)
    ax_mem.grid(True, axis='x', alpha=0.3)

    _savefig(fig, 'benchmark_cuda')


# ---------------------------------------------------------------------------
# Figure 5 — large-scale hidden sweep (batch=512, seq_len=100)
# ---------------------------------------------------------------------------

def plot_largescale(all_records):
    """
    Hidden size swept from 64→2048 at batch=512. Two panels:
      Left:  log-scale step time vs hidden size (JAX CPU, Torch CPU, Torch MPS, Torch CUDA)
      Right: speedup over PyTorch CPU at each hidden size
    """
    BATCH, SEQ = 512, 500
    hiddens = [64, 128, 256, 512, 1024, 2048]

    combos = [
        ('jax',   'cpu',  STYLE[('jax',   'cpu')]),
        ('jax',   'cuda', STYLE[('jax',   'cuda')]),
        ('torch', 'cpu',  STYLE[('torch', 'cpu')]),
        ('torch', 'gpu',  STYLE[('torch', 'gpu')]),
        ('torch', 'cuda', STYLE[('torch', 'cuda')]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f'Large-scale hidden sweep  (batch={BATCH}, seq_len={SEQ}, vanilla RNN)',
        fontsize=12, fontweight='bold')

    # ── Left: absolute step times (log y) ────────────────────────────────
    ax = axes[0]
    for fw, dev, sty in combos:
        recs = select(all_records, framework=fw, device=dev,
                      model='vanilla', seq_len=SEQ, batch=BATCH)
        if not recs:
            continue
        recs = sorted(recs, key=lambda r: r['hidden'])
        xs  = [r['hidden']     for r in recs]
        ys  = [r['median_ms']  for r in recs]
        lo  = [r['median_ms'] - r['min_ms']  for r in recs]
        hi  = [r['max_ms']    - r['median_ms'] for r in recs]
        ax.errorbar(xs, ys, yerr=[lo, hi],
                    color=sty['color'], ls=sty['ls'], marker=sty['marker'],
                    ms=6, lw=2, capsize=3, label=sty['label'])
        ax.fill_between(xs,
                         [y - l for y, l in zip(ys, lo)],
                         [y + h for y, h in zip(ys, hi)],
                         color=sty['color'], alpha=ALPHA_BAND)

    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xticks(hiddens)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Hidden size', fontsize=10)
    ax.set_ylabel('Median step time (ms, log scale)', fontsize=10)
    ax.set_title('A  Step time vs hidden size', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)

    # Annotate crossover where MPS beats CPU
    mps_recs   = {r['hidden']: r for r in select(all_records, framework='torch',
                  device='gpu', model='vanilla', seq_len=SEQ, batch=BATCH)}
    cpu_recs   = {r['hidden']: r for r in select(all_records, framework='torch',
                  device='cpu', model='vanilla', seq_len=SEQ, batch=BATCH)}
    crossovers = [h for h in hiddens
                  if h in mps_recs and h in cpu_recs
                  and mps_recs[h]['median_ms'] < cpu_recs[h]['median_ms']]
    if crossovers:
        x0 = crossovers[0]
        ax.axvline(x0, color='grey', lw=1, ls=':', alpha=0.7)
        ax.text(x0 * 1.05, ax.get_ylim()[0] * 1.5,
                f'MPS < CPU\n(h≥{x0})', fontsize=7.5, color='grey', va='bottom')

    # ── Right: speedup over PyTorch CPU ──────────────────────────────────
    ax2 = axes[1]
    ref = {r['hidden']: r['median_ms']
           for r in select(all_records, framework='torch', device='cpu',
                           model='vanilla', seq_len=SEQ, batch=BATCH)}
    if not ref:
        ax2.set_visible(False)
    else:
        speedup_combos = [
            ('jax',   'cpu',  STYLE[('jax',   'cpu')],  'JAX CPU / PyTorch CPU'),
            ('torch', 'gpu',  STYLE[('torch', 'gpu')],  'PyTorch MPS / PyTorch CPU'),
            ('torch', 'cuda', STYLE[('torch', 'cuda')], 'PyTorch CUDA / PyTorch CPU'),
            ('jax',   'cuda', STYLE[('jax',   'cuda')], 'JAX CUDA / PyTorch CPU'),
        ]
        x = np.arange(len(hiddens))
        bw = 0.18
        offsets = np.linspace(-(len(speedup_combos)-1)/2, (len(speedup_combos)-1)/2,
                              len(speedup_combos)) * bw

        for (fw, dev, sty, lbl), offset in zip(speedup_combos, offsets):
            recs_d = {r['hidden']: r['median_ms']
                      for r in select(all_records, framework=fw, device=dev,
                                      model='vanilla', seq_len=SEQ, batch=BATCH)}
            ratios = [ref.get(h, None) / recs_d.get(h, None)
                      if (h in ref and h in recs_d) else None
                      for h in hiddens]
            valid_x = [x[i] + offset for i, v in enumerate(ratios) if v is not None]
            valid_r = [v for v in ratios if v is not None]
            if valid_r:
                ax2.bar(valid_x, valid_r, width=bw * 0.9,
                        color=sty['color'], label=lbl, alpha=0.85)

        ax2.axhline(1.0, color='grey', lw=1, ls='--')
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(h) for h in hiddens])
        ax2.set_xlabel('Hidden size', fontsize=10)
        ax2.set_ylabel('Speedup over PyTorch CPU  (higher = faster)', fontsize=10)
        ax2.set_title('B  Speedup over PyTorch CPU', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Note about pending CUDA data
    cuda_present = any(r['device'] == 'cuda' and r['batch'] == BATCH
                       and r['model'] == 'vanilla' and r['seq_len'] == SEQ
                       for r in all_records)
    if not cuda_present:
        fig.text(0.5, -0.03,
                 'CUDA results at batch=512 pending — run on remote GPU server:\n'
                 'for h in 64 128 256 512 1024 2048; do python benchmarks/run_benchmarks.py '
                 '--device cuda --framework both --model vanilla --hidden $h --seq-len 100 --batch 512; done',
                 ha='center', fontsize=7.5, color='grey', style='italic',
                 transform=fig.transFigure)

    plt.tight_layout()
    _savefig(fig, 'benchmark_largescale')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_label(k):
    model, hidden, seq_len, batch, rank = k
    s = f'{model}  h={hidden}  T={seq_len}'
    if rank is not None:
        s += f'  r={rank}'
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    records = load_results()
    by_dev = {}
    for r in records:
        by_dev[r['device']] = by_dev.get(r['device'], 0) + 1
    print(f'Loaded {len(records)} records: ' +
          '  '.join(f'{d}={n}' for d, n in sorted(by_dev.items())))

    plot_sweeps(records)
    plot_speedup(records)
    plot_largescale(records)
    plot_cpu_vs_mps(records)
    plot_cuda(records)
    print('Done.')
