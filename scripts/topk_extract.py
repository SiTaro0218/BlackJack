"""Extract top-K parameter combinations by avg_reward from a results CSV.
Saves CSV and a bar plot under figures/.
"""
import os
import sys
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt

TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def main(path, k):
    df = pd.read_csv(path)
    ensure_dir('figures')

    # determine score column
    score_col = None
    for c in ['avg_reward','avg_episode_reward','total_reward','reward','score','mean_reward']:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        print('No score column found in', path)
        return 2

    # group columns to consider for uniqueness
    group_cols = [c for c in ['alpha','gamma','bins','eps_decay_type','eps_start','eps_end','seed'] if c in df.columns]
    # compute mean and mov100 mean per parameter combination excluding seed if present
    agg_by = [c for c in group_cols if c != 'seed']

    grouped = df.groupby(agg_by).agg(
        runs=('run_name','count'),
        mean_score=(score_col,'mean'),
        median_score=(score_col,'median'),
        std_score=(score_col,'std')
    ).reset_index()

    topk = grouped.sort_values('mean_score', ascending=False).head(k)

    out_csv = os.path.join('figures', f'top{k}_by_mean_{TS}.csv')
    topk.to_csv(out_csv, index=False)
    print('Saved top-k CSV to', out_csv)

    # bar plot
    plt.figure(figsize=(10,5))
    labels = topk.apply(lambda row: '\n'.join([f"{c}={row[c]}" for c in agg_by]), axis=1)
    plt.bar(range(len(topk)), topk['mean_score'], yerr=topk['std_score'].fillna(0), capsize=4)
    plt.xticks(range(len(topk)), labels, rotation=45, ha='right')
    plt.ylabel('mean_'+score_col)
    plt.title(f'Top {k} param combos by mean {score_col}')
    plt.tight_layout()
    out_png = os.path.join('figures', f'top{k}_by_mean_{TS}.png')
    plt.savefig(out_png)
    plt.close()
    print('Saved bar plot to', out_png)

    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='results_autocollect_FULL_20251110_121110.csv')
    p.add_argument('--k', type=int, default=10)
    args = p.parse_args()
    if not os.path.exists(args.path):
        print('File not found:', args.path)
        sys.exit(2)
    sys.exit(main(args.path, args.k))
