# Quick analysis for results CSV
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt

TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def main(path):
    df = pd.read_csv(path)
    ensure_dir('figures')

    # Prefer avg_reward and mov_avg_100 if present
    score_col = None
    for c in ['avg_reward','avg_episode_reward','total_reward','reward','score','mean_reward']:
        if c in df.columns:
            score_col = c
            break
    mov100_col = 'mov_avg_100' if 'mov_avg_100' in df.columns else None

    # Basic stats
    stats = {}
    if score_col is not None:
        s = df[score_col].dropna()
        stats['col'] = score_col
        stats['count'] = int(s.count())
        stats['mean'] = float(s.mean())
        stats['median'] = float(s.median())
        stats['std'] = float(s.std())
        stats['min'] = float(s.min())
        stats['max'] = float(s.max())
        stats['25q'] = float(s.quantile(0.25))
        stats['75q'] = float(s.quantile(0.75))

    if mov100_col is not None:
        m = df[mov100_col].dropna()
        stats['mov100_mean'] = float(m.mean())
        stats['mov100_median'] = float(m.median())

    # Save overall stats
    summary_path = os.path.join('figures', f'summary_overall_{TS}.csv')
    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    print('Saved overall summary to', summary_path)

    # Grouped summaries
    group_cols = ['alpha','gamma','bins','eps_decay_type']
    available_groups = [c for c in group_cols if c in df.columns]
    if score_col is None:
        print('No obvious score column found. Exiting grouped analysis.')
        return

    # aggregated means per group
    agg = df.groupby(available_groups)[score_col].agg(['count','mean','median','std','min','max']).reset_index()
    agg_path = os.path.join('figures', f'summary_by_params_{TS}.csv')
    agg.to_csv(agg_path, index=False)
    print('Saved aggregated summary to', agg_path)

    # Simple plots
    try:
        plt.figure(figsize=(6,4))
        if 'alpha' in df.columns:
            a = df.groupby('alpha')[score_col].mean().sort_index()
            a.plot(kind='bar')
            plt.ylabel('mean_'+score_col)
            plt.title('mean {} by alpha'.format(score_col))
            p = os.path.join('figures', f'avg_by_alpha_{TS}.png')
            plt.tight_layout(); plt.savefig(p); plt.close()
            print('Saved', p)

        if 'gamma' in df.columns:
            plt.figure(figsize=(6,4))
            g = df.groupby('gamma')[score_col].mean().sort_index()
            g.plot(kind='bar')
            plt.ylabel('mean_'+score_col)
            plt.title('mean {} by gamma'.format(score_col))
            p = os.path.join('figures', f'avg_by_gamma_{TS}.png')
            plt.tight_layout(); plt.savefig(p); plt.close()
            print('Saved', p)

        if 'bins' in df.columns:
            plt.figure(figsize=(6,4))
            b = df.groupby('bins')[score_col].mean().sort_values()
            b.plot(kind='bar')
            plt.ylabel('mean_'+score_col)
            plt.title('mean {} by bins'.format(score_col))
            p = os.path.join('figures', f'avg_by_bins_{TS}.png')
            plt.tight_layout(); plt.savefig(p); plt.close()
            print('Saved', p)

        # alpha x gamma heatmap if both exist
        if 'alpha' in df.columns and 'gamma' in df.columns:
            pivot = df.pivot_table(index='alpha', columns='gamma', values=score_col, aggfunc='mean')
            plt.figure(figsize=(8,6))
            im = plt.imshow(pivot.values, aspect='auto', cmap='viridis')
            plt.colorbar(im)
            plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45)
            plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            plt.title('mean {} (alpha x gamma)'.format(score_col))
            p = os.path.join('figures', f'alpha_gamma_heatmap_{TS}.png')
            plt.tight_layout(); plt.savefig(p); plt.close()
            print('Saved', p)

        # histogram of scores
        plt.figure(figsize=(6,4))
        df[score_col].hist(bins=40)
        plt.title(f'Histogram of {score_col}')
        p = os.path.join('figures', f'hist_{score_col}_{TS}.png')
        plt.tight_layout(); plt.savefig(p); plt.close()
        print('Saved', p)
    except Exception as e:
        print('Plotting failed:', e)

    print('All done.')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results_autocollect_FULL_20251110_121110.csv'
    if not os.path.exists(path):
        print('File not found:', path)
        sys.exit(2)
    main(path)
