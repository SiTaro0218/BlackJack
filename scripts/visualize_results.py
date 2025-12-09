#!/usr/bin/env python3
"""Visualize results.csv and logs produced by run_experiments.py

Produces:
 - figures/summary_bar.png        : bar chart of mov_avg_100 per run
 - figures/top_learning_curves.png: learning curves (moving avg) for top runs

Usage: python visualize_results.py
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from collections import OrderedDict


def parse_rewards_from_log_text(text):
    pattern = re.compile(r'total reward:\s*([0-9]+(?:\.[0-9]+)?)')
    return [float(m.group(1)) for m in pattern.finditer(text)]


def parse_rewards_from_history_file(history_path):
    """Parse ai_player_Q history CSV (score,hand_length,action,result,reward) into per-episode total rewards."""
    if not os.path.exists(history_path):
        return []
    rewards = []
    cur_sum = 0.0
    terminal_statuses = set(['bust', 'win', 'lose', 'surrender'])
    try:
        with open(history_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                # expected: score,hand_length,action,result,reward
                try:
                    status = row[3].strip().lower()
                except Exception:
                    status = ''
                try:
                    r = float(row[4])
                except Exception:
                    r = 0.0
                cur_sum += r
                if status in terminal_statuses:
                    rewards.append(cur_sum)
                    cur_sum = 0.0
    except Exception:
        return []
    return rewards


def moving_average(arr, window=100):
    # return moving average array of same length as input
    n = len(arr)
    if n == 0:
        return np.array([])
    a = np.array(arr, dtype=float)
    if window <= 1:
        return a
    cumsum = np.cumsum(np.insert(a, 0, 0.0))
    mov = np.empty(n, dtype=float)
    for i in range(n):
        denom = min(window, i+1)
        mov[i] = (cumsum[i+1] - cumsum[i+1-denom]) / float(denom)
    return mov


def read_results(csv_path='results.csv'):
    if not os.path.exists(csv_path):
        print('results.csv not found at', csv_path)
        sys.exit(1)
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    rows = read_results('results.csv')
    if not rows:
        print('No results in results.csv')
        return

    # collect data rows with parsed numeric metrics
    entries = []
    for r in rows:
        try:
            mov = float(r.get('mov_avg_100', 0))
        except:
            mov = np.nan
        try:
            avg = float(r.get('avg_reward', 0))
        except:
            avg = np.nan
        logfile = r.get('logfile', '')
        # parse hyperparams to form a group key
        key = (r.get('alpha',''), r.get('gamma',''), r.get('eps_decay_type',''), r.get('bins',''), r.get('eps_start',''), r.get('eps_end',''))
        entries.append({'row': r, 'mov': mov, 'avg': avg, 'logfile': logfile, 'group_key': key})

    # group by hyperparam key
    groups = OrderedDict()
    for e in entries:
        k = e['group_key']
        groups.setdefault(k, []).append(e)

    ensure_dir('figures')

    # Aggregate summary per group
    summary = []
    for k, items in groups.items():
        movs = np.array([it['mov'] for it in items], dtype=float)
        avgs = np.array([it['avg'] for it in items], dtype=float)
        summary.append({'group_key': k, 'n_runs': len(items), 'mov_mean': np.nanmean(movs), 'mov_std': np.nanstd(movs), 'avg_mean': np.nanmean(avgs), 'avg_std': np.nanstd(avgs), 'logs': [it['logfile'] for it in items]})

    # sort groups by mov_mean desc
    summary_sorted = sorted(summary, key=lambda x: x['mov_mean'] if not np.isnan(x['mov_mean']) else -np.inf, reverse=True)

    # write summary CSV
    summary_csv = os.path.join('figures', 'summary_table.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha','gamma','eps_decay_type','bins','eps_start','eps_end','n_runs','mov_mean','mov_std','avg_mean','avg_std'])
        for s in summary_sorted:
            a,g,dt,b,es,ee = s['group_key']
            writer.writerow([a,g,dt,b,es,ee,s['n_runs'],s['mov_mean'],s['mov_std'],s['avg_mean'],s['avg_std']])
    print('Wrote', summary_csv)

    # Bar chart: top groups by mov_mean
    top_groups = min(20, len(summary_sorted))
    labels = []
    values = []
    errors = []
    for s in summary_sorted[:top_groups]:
        a,g,dt,b,es,ee = s['group_key']
        labels.append(f"a={a},g={g},{dt},{b}")
        values.append(s['mov_mean'])
        errors.append(s['mov_std'])

    plt.figure(figsize=(max(10, len(labels)*0.5), 6))
    x = np.arange(len(labels))
    plt.bar(x, values, yerr=errors, capsize=4)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=8)
    plt.ylabel('mov_avg_100 (mean ± std)')
    plt.title('Top groups by mov_avg_100 (grouped by hyperparams)')
    plt.tight_layout()
    out1 = os.path.join('figures', 'summary_bar_groups.png')
    plt.savefig(out1, dpi=200)
    plt.close()
    print('Wrote', out1)

    # For top N groups, plot mean learning curve with shaded std
    N = min(6, len(summary_sorted))
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    for i in range(N):
        s = summary_sorted[i]
        logs = s['logs']
        # collect rewards arrays for each run in group
        rewards_list = []
        max_len = 0
        for log in logs:
            # First try history CSV associated with the log (ai_player_Q)
            hist_path = None
            try:
                if log.endswith('.txt'):
                    hist_path = log.replace('.txt', '.history.csv')
                else:
                    hist_path = log + '.history.csv'
                if not os.path.exists(hist_path):
                    # try in logs directory
                    base = os.path.basename(log)
                    name = os.path.splitext(base)[0]
                    hist_path = os.path.join('logs', name + '.history.csv')
            except Exception:
                hist_path = None

            r = []
            if hist_path and os.path.exists(hist_path):
                r = parse_rewards_from_history_file(hist_path)
            else:
                try:
                    with open(log, encoding='utf-8') as f:
                        txt = f.read()
                except:
                    # try relative
                    try:
                        with open(os.path.join('.', log), encoding='utf-8') as f:
                            txt = f.read()
                    except Exception as e:
                        print('Failed to read log', log, e)
                        continue
                r = parse_rewards_from_log_text(txt)
            if len(r) > 0:
                rewards_list.append(r)
                max_len = max(max_len, len(r))
        if not rewards_list:
            continue
        # build matrix with NaN padding
        mat = np.full((len(rewards_list), max_len), np.nan)
        for idx, arr in enumerate(rewards_list):
            mat[idx, :len(arr)] = arr
        # compute moving average per-run, then mean/std across runs
        movs = np.array([moving_average(row.tolist(), window=100) for row in mat])
        mean_mov = np.nanmean(movs, axis=0)
        std_mov = np.nanstd(movs, axis=0)
        episodes = np.arange(1, len(mean_mov)+1)
        label = f"{s['group_key'][0]} / {s['group_key'][1]} / {s['group_key'][2]} / {s['group_key'][3]}"
        c = colors[i % len(colors)]
        plt.plot(episodes, mean_mov, label=label, color=c)
        plt.fill_between(episodes, mean_mov - std_mov, mean_mov + std_mov, color=c, alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Moving average reward (window=100)')
    plt.title('Top group learning curves (mean ± std across runs)')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    out2 = os.path.join('figures', 'group_learning_curves.png')
    plt.savefig(out2, dpi=200)
    plt.close()
    print('Wrote', out2)


if __name__ == '__main__':
    main()
