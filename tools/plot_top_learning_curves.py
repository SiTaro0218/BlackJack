import os
import csv
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
SUMMARY = os.path.join(BASE, 'figures', 'summary_table.csv')
LOGDIR = os.path.join(BASE, 'logs')
OUTDIR = os.path.join(BASE, 'figures')
os.makedirs(OUTDIR, exist_ok=True)

def norm_val(v):
    s = str(v)
    return s.replace('.', '_')

# parse summary and pick top N by metric
def pick_top_n(n=3, metric='mov_mean'):
    rows = []
    with open(SUMMARY, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row[metric] = float(row.get(metric, 'nan'))
            except:
                row[metric] = float('nan')
            rows.append(row)
    rows_sorted = sorted(rows, key=lambda x: x[metric], reverse=True)
    return rows_sorted[:n]

# find history files matching a summary row
def find_matching_histories(row):
    # create substrings to match in filename
    keys = ['alpha','gamma','eps_start','eps_end','eps_decay_type']
    substrs = []
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        substrs.append(f"{k}-{norm_val(v)}")
    # search
    matches = []
    for p in glob.glob(os.path.join(LOGDIR, '*.history.csv')):
        name = os.path.basename(p)
        ok = True
        for s in substrs:
            if s not in name:
                ok = False; break
        if ok:
            matches.append(p)
    return sorted(matches)

# parse history file into episode rewards
def parse_history_rewards(path):
    rewards = []
    cur_sum = 0.0
    terminal_results = set(['win','lose','surrendered','blackjack','bust'])
    try:
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            r = csv.DictReader(f)
            for row in r:
                # reward column
                try:
                    rew = float(row.get('reward', 0))
                except:
                    try:
                        rew = float(row.get('reward', 0) or 0)
                    except:
                        rew = 0.0
                cur_sum += rew
                res = (row.get('result') or '').strip().lower()
                if res and res not in ['', 'unsettled', 'retry']:
                    # consider this the end of episode
                    rewards.append(cur_sum)
                    cur_sum = 0.0
    except Exception as e:
        print('Failed to parse', path, e)
    # if leftover
    if cur_sum != 0.0:
        rewards.append(cur_sum)
    return rewards

# moving average
def movavg(a, w=50):
    if not a:
        return []
    if w <= 1:
        return a
    out = []
    s = 0.0
    for i,x in enumerate(a):
        s += x
        if i >= w:
            s -= a[i-w]
            out.append(s/w)
        else:
            out.append(s/(i+1))
    return out


def main(n=3, metric='mov_mean'):
    top = pick_top_n(n=n, metric=metric)
    if not top:
        print('No top rows found in', SUMMARY)
        return
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    plotted = []
    for i,row in enumerate(top):
        matches = find_matching_histories(row)
        if not matches:
            print('No history files for group:', row)
            continue
        # plot up to 5 files per group
        group_rewards = []
        for p in matches[:5]:
            r = parse_history_rewards(p)
            if not r:
                continue
            group_rewards.append(r)
            mov = movavg(r, w=50)
            ax.plot(range(1, len(mov)+1), mov, color=colors[i%len(colors)], alpha=0.35)
        # plot mean of runs if multiple
        if group_rewards:
            # pad to same length
            maxlen = max(len(x) for x in group_rewards)
            padded = []
            for x in group_rewards:
                if len(x) < maxlen:
                    x = x + [x[-1]]*(maxlen-len(x))
                padded.append(x)
            mean_series = [statistics.mean(col) for col in zip(*padded)]
            mov_mean_series = movavg(mean_series, w=50)
            ax.plot(range(1, len(mov_mean_series)+1), mov_mean_series, color=colors[i%len(colors)], linewidth=2.5, label=f"{row.get('alpha')}, {row.get('gamma')}, {row.get('eps_decay_type')}, es={row.get('eps_start')}, ee={row.get('eps_end')}")
            plotted.append((row, matches))
    ax.set_xlabel('Episode (moving avg window=50)')
    ax.set_ylabel('Moving average reward')
    ax.set_title(f'Top {n} groups by {metric} - learning curves')
    ax.legend()
    out = os.path.join(OUTDIR, f'top{n}_learning_curves_{metric}.png')
    fig.tight_layout()
    fig.savefig(out)
    print('Wrote', out)
    # print plotted groups summary
    for row, matches in plotted:
        print('Plotted group:', row.get('alpha'), row.get('gamma'), row.get('eps_decay_type'), 'matches:', len(matches))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--metric', default='mov_mean')
    args = p.parse_args()
    main(n=args.n, metric=args.metric)
