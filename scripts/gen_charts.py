#!/usr/bin/env python3
"""Generate benchmark charts for the repo README. Data = measured values from the
tuning session (see docs/tuning-record). Run: python3 gen_charts.py"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'images'  # repo-relative output dir
plt.rcParams.update({'figure.dpi': 140, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})

# ---------- Chart 1: decode TPS vs context length ----------
ctx = [0.1, 5, 20, 50, 100]
nospec = [73.5, 73.5, 72.1, 70.1, 68.0]           # measured, nospec flashinfer
eagle_ctx = [0.5, 50]
eagle = [142.0, 127.0]                            # short: 139-144 measured; 50k: median of 107-148
eagle_err = [[2.0, 20.0], [2.0, 21.0]]            # down/up error bars

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ctx, nospec, 'o--', color='#888', lw=1.8, label='No speculation (flashinfer)')
ax.errorbar(eagle_ctx, eagle, yerr=eagle_err, fmt='o-', color='#d62728', lw=2.2,
            capsize=4, label='EAGLE MTP 3/1/4 (final)')
ax.set_xscale('symlog')
ax.set_xticks([0.1, 1, 5, 20, 50, 100])
ax.set_xticklabels(['0.1k', '1k', '5k', '20k', '50k', '100k'])
ax.set_xlabel('Input context length (tokens)')
ax.set_ylabel('Single-stream decode (tokens/s)')
ax.set_title('Decode throughput vs context length\nQwen3.8-27B-FP8, TP4 on 4x RTX 4090, SGLang 0.5.17')
ax.legend(loc='lower left')
ax.annotate('1.9x @ short\n1.5-2.1x @ 50k', xy=(50, 127), xytext=(28, 95),
            arrowprops=dict(arrowstyle='->', color='#444'), fontsize=9, color='#222')
fig.tight_layout()
fig.savefig(f'{OUT}/tps_vs_context.png')
plt.close(fig)

# ---------- Chart 2: decode-step kernel time breakdown ----------
labels = ['W8A8 FP8 GEMM', 'NCCL all-reduce\n(4x PCIe, no P2P)', 'GEMV (lm_head\n+ ba proj)', 'Quant + RMSNorm', 'GDN fused kernels\n(conv1d + packed decode)', 'Other']
times = [8.6, 3.0, 1.1, 0.9, 0.29, 0.21]  # ms per decode step, torch profiler, total 14.1ms
colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c', '#ff7f0e', '#bbb']

fig, ax = plt.subplots(figsize=(7.2, 4.0))
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, times, color=colors, height=0.62)
ax.set_yticks(y, labels)
ax.set_xlabel('ms per decode step (lower = better)')
ax.set_title('Decode step kernel breakdown: 14.1 ms/step total\nGDN linear attention is already fused - only ~2%; NCCL is the real tax (21%)')
for yi, t in zip(y, times):
    ax.text(t + 0.08, yi, f'{t:.2f} ms ({t/14.1*100:.0f}%)', va='center', fontsize=9)
ax.set_xlim(0, 10.2)
fig.tight_layout()
fig.savefig(f'{OUT}/decode_kernel_breakdown.png')
plt.close(fig)

# ---------- Chart 3: EAGLE parameter sweep ----------
cfgs = ['no spec', 'steps=3\ntopk=1\ndraft=4\n(final)', 'steps=4\ntopk=1\ndraft=5', 'steps=3\ntopk=2\ndraft=6']
tps = [74.85, 142.2, 144.4, 119.5]
acc = [1.0, 2.92, 3.31, 2.86]
x = np.arange(len(cfgs))
fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
b = ax1.bar(x, tps, width=0.52, color=['#888', '#d62728', '#ff9896', '#9467bd'])
for xi, v in zip(x, tps):
    ax1.text(xi, v + 2, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
ax1.set_xticks(x, cfgs)
ax1.set_ylabel('Short-context decode (tokens/s)')
ax1.set_ylim(0, 165)
ax2 = ax1.twinx()
ax2.plot(x, acc, 'o-', color='#1f77b4', lw=2, label='accept length')
for xi, v in zip(x, acc):
    ax2.annotate(f'{v:.2f}', (xi, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9, color='#1f77b4')
ax2.set_ylabel('EAGLE accept length (tokens/iter)', color='#1f77b4')
ax2.set_ylim(0, 4.2)
ax2.spines['right'].set_visible(True)
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_title('EAGLE speculative decoding parameter sweep (short context)')
fig.tight_layout()
fig.savefig(f'{OUT}/eagle_sweep.png')
plt.close(fig)

# ---------- Chart 4: prefill TTFT ----------
lens = [65, 131]
ttft_lo = [14.25, 32.23]   # best measured
ttft_hi = [16.21, 34.03]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.bar(['65k tokens', '131k tokens'], [np.mean([14.25, 16.21]), np.mean([32.23, 34.03])],
       yerr=[[0, 0], [16.21-14.25, 34.03-32.23]], width=0.4, color='#2ca02c', capsize=5)
ax.text(0, 15.6, '14.3-16.2 s\n(~4.1-4.6k tok/s)', ha='center', fontsize=9)
ax.text(1, 33.6, '32.2-34.0 s\n(~3.8-4.1k tok/s)', ha='center', fontsize=9)
ax.set_ylabel('TTFT, cache miss (s)')
ax.set_ylim(0, 40)
ax.set_title('Long-context prefill TTFT (GEMM-bound, ~70% of 4x4090 FP8 peak)\nchunk 8k = 16k; chunk 32k OOMs with EAGLE')
fig.tight_layout()
fig.savefig(f'{OUT}/prefill_ttft.png')
plt.close(fig)

print('charts written')

# ---------- Chart: HiCache host-memory prefix reload (2026-08-17) ----------
fig, ax = plt.subplots(figsize=(6.6, 3.9))
labels = ['Cold prefill\n(GPU recompute)', 'HiCache reload\n(host RAM L2)']
vals = [10.86, 1.96]  # measured TTFT, 31.6k-token needle prefix after forced eviction
bars = ax.bar(labels, vals, color=['#888888', '#2ca02c'], width=0.52)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f'{v:.2f} s', ha='center', fontsize=11)
ax.annotate('5.5x faster\nneedle answer correct after reload', xy=(1, 1.96), xytext=(0.62, 6.2),
            arrowprops=dict(arrowstyle='->', color='#444'), fontsize=9.5, color='#222')
ax.set_ylabel('TTFT (seconds, lower = better)')
ax.set_title('HiCache L2 (host RAM) prefix reload, 31.6k-token prefix\n'
             'evicted from GPU by 975k-token pressure, then re-requested\n'
             'Qwen3.8-27B-FP8 (hybrid GDN), 4x RTX 4090, SGLang 0.5.17')
ax.set_ylim(0, 12.5)
fig.tight_layout()
fig.savefig(f'{OUT}/hicache_reload.png')
plt.close(fig)
print('hicache_reload.png written')
