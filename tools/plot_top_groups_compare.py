import csv
import os
import argparse
from collections import defaultdict
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_results(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast some fields
            r['episodes'] = int(r.get('episodes', '0') or 0)
            try:
                r['mov_avg_100'] = float(r.get('mov_avg_100', '0') or 0)
            except Exception:
                r['mov_avg_100'] = 0.0
            rows.append(r)
    return rows


def find_history_file(run_name, logs_dir='logs'):
    # try several common suffixes
    candidates = [f"{run_name}.history.csv", f"{run_name}.long.history.csv", f"{run_name}.short.history.csv"]
    for c in candidates:
        p = os.path.join(logs_dir, c)
        if os.path.exists(p):
            return p
    # fallback: try glob-like search in logs
    for fname in os.listdir(logs_dir):
        if fname.startswith(run_name) and fname.endswith('.history.csv'):
            return os.path.join(logs_dir, fname)
    return None


def load_episode_rewards(history_path):
    rewards = []
    with open(history_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # expect a column named 'reward' and 'result' (unsettled for intermediate)
        for row in reader:
            res = row.get('result', '').strip()
            if res == 'unsettled':
                continue
            # final step for a hand; take reward
            try:
                r = float(row.get('reward', '0') or 0)
            except Exception:
                try:
                    r = float(row.get('reward', '0').replace('"', ''))
                except Exception:
                    r = 0.0
            rewards.append(r)
    return rewards


def moving_average(x, w=100):
    if len(x) == 0:
        return np.array([])
    a = np.array(x, dtype=float)
    if w <= 1:
        return a
    ret = np.convolve(a, np.ones(w, dtype=float), 'valid') / w
    # pad front so length == len(a)
    pad = np.full(w-1, np.nan)
    return np.concatenate([pad, ret])


def group_key_from_row(r):
    return (r['alpha'], r['gamma'], r['eps_start'], r['eps_end'], r['eps_decay_type'])


def pretty_group_name(key):
    alpha, gamma, eps_start, eps_end, decay = key
    return f"a={alpha} g={gamma} eps={eps_start}->{eps_end} {decay}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results.csv')
    p.add_argument('--logs', default='logs')
    p.add_argument('--top', type=int, default=3)
    p.add_argument('--min_episodes', type=int, default=900)
    p.add_argument('--out', default='figures/top_groups_compare.png')
    args = p.parse_args()

    rows = read_results(args.results)

    # filter long runs
    long_rows = [r for r in rows if r['episodes'] >= args.min_episodes]
    if not long_rows:
        print('No runs with episodes >=', args.min_episodes)
        return

    # aggregate by group (ignore seed)
    groups = defaultdict(list)
    for r in long_rows:
        key = group_key_from_row(r)
        groups[key].append(r)

    # compute mean mov_avg_100 across seeds for ranking (only groups with >=2 seeds)
    group_scores = []
    for k, rs in groups.items():
        vals = [r['mov_avg_100'] for r in rs]
        if len(vals) >= 2:
            group_scores.append((np.mean(vals), k, rs))
    if not group_scores:
        print('No groups with >=2 seeds and enough episodes')
        return

    # lower mov_avg_100 is better? In these logs negative numbers; we sort ascending
    group_scores.sort(key=lambda t: t[0])
    top = group_scores[:args.top]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(10, 6))

    for mean_score, key, rs in top:
        # for each seed row in the group, load history and compute mov_avg
        series = []
        seeds_used = []
        for r in rs:
            run_name = r['run_name']
            hist = find_history_file(run_name, args.logs)
            if not hist:
                print('missing history for', run_name)
                continue
            rewards = load_episode_rewards(hist)
            if len(rewards) < 10:
                print('too-short history', run_name, 'len', len(rewards))
                continue
            ma = moving_average(rewards, w=100)
            series.append(ma)
            seeds_used.append(r['seed'])

        if not series:
            print('no valid histories for group', key)
            continue

        # truncate to shortest length (to align)
        minlen = min(len(s) for s in series)
        data = np.vstack([s[:minlen] for s in series])
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        x = np.arange(len(mean))
        label = pretty_group_name(key)
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('MovAvg100 reward')
    plt.title('Top parameter groups comparison (mean ± std across seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
