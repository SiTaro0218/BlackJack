import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
results_csv = os.path.join(ROOT, 'results_autocollect_FULL_20251110_121110.csv')
fig_dir = os.path.join(ROOT, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Read
df = pd.read_csv(results_csv)
# ensure numeric
for c in ['alpha','gamma','eps_start','eps_end','episodes','avg_reward','mov_avg_100']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Summary by parameter
group_cols = ['alpha','gamma','bins','eps_decay_type']
summary = []
for col in ['alpha','gamma','bins','eps_decay_type']:
    g = df.groupby(col)['avg_reward'].agg(['mean','std','count']).reset_index()
    g.columns = [col,'mean_avg_reward','std_avg_reward','count']
    g.to_csv(os.path.join(fig_dir, f'summary_by_{col}.csv'), index=False)
    summary.append((col, g))

# Save combined summary
combined = []
for col, g in summary:
    g['param']=col
    g = g.rename(columns={col:'value'})
    combined.append(g[['param','value','mean_avg_reward','std_avg_reward','count']])
combined_df = pd.concat(combined, ignore_index=True)
combined_df.to_csv(os.path.join(fig_dir,'summary_by_param_combined.csv'), index=False)

# Plot mean +/- std for numeric categorical params
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn' in plt.style.available else 'ggplot')

# alpha
if 'alpha' in df.columns:
    g = df.groupby('alpha')['avg_reward'].agg(['mean','std']).reset_index()
    plt.figure(figsize=(6,4))
    plt.errorbar(g['alpha'], g['mean'], yerr=g['std'], fmt='o-', capsize=5)
    plt.xlabel('alpha')
    plt.ylabel('mean avg_reward')
    plt.title('Mean avg_reward by alpha')
    plt.savefig(os.path.join(fig_dir,'avg_by_alpha.png'), bbox_inches='tight')
    plt.close()

# gamma
if 'gamma' in df.columns:
    g = df.groupby('gamma')['avg_reward'].agg(['mean','std']).reset_index()
    plt.figure(figsize=(6,4))
    plt.errorbar(g['gamma'], g['mean'], yerr=g['std'], fmt='o-', capsize=5)
    plt.xlabel('gamma')
    plt.ylabel('mean avg_reward')
    plt.title('Mean avg_reward by gamma')
    plt.savefig(os.path.join(fig_dir,'avg_by_gamma.png'), bbox_inches='tight')
    plt.close()

# bins (categorical)
if 'bins' in df.columns:
    g = df.groupby('bins')['avg_reward'].agg(['mean','std']).reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(g['bins'], g['mean'], yerr=g['std'], capsize=5)
    plt.xlabel('bins')
    plt.ylabel('mean avg_reward')
    plt.title('Mean avg_reward by bins')
    plt.savefig(os.path.join(fig_dir,'avg_by_bins.png'), bbox_inches='tight')
    plt.close()

# heatmap alpha x gamma
if set(['alpha','gamma']).issubset(df.columns):
    pivot = df.pivot_table(values='avg_reward', index='alpha', columns='gamma', aggfunc='mean')
    plt.figure(figsize=(6,5))
    im = plt.imshow(pivot.values, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='mean avg_reward')
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.xlabel('gamma')
    plt.ylabel('alpha')
    plt.title('Mean avg_reward (alpha x gamma)')
    plt.savefig(os.path.join(fig_dir,'alpha_gamma_heatmap.png'), bbox_inches='tight')
    plt.close()

print('Saved figures to', fig_dir)
print('Saved combined summary to', os.path.join(fig_dir,'summary_by_param_combined.csv'))
