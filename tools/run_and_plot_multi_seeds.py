import os, glob, subprocess, sys, shlex, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
QT_DIR = os.path.join(BASE, 'logs', 'qtables')
PATTERN = os.path.join(QT_DIR, 'alpha-0_1--gamma-0_9--eps_start-1_0--eps_end-0_05--eps_decay_type-const--seed-*.pkl')
OUT_FIG = os.path.join(BASE, 'figures', 'top1_multi_seed_money.png')
LOGDIR = os.path.join(BASE, 'logs')
os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)

# find qtables
qt_files = sorted(glob.glob(PATTERN))
if not qt_files:
    print('No qtable files matching pattern:', PATTERN)
    sys.exit(2)
print('Found qtables:', qt_files)

histories = []
finals = []

# run each qtable sequentially
for q in qt_files:
    base = os.path.splitext(os.path.basename(q))[0]
    hist = os.path.join(LOGDIR, base + '.long.history.csv')
    outlog = os.path.join(LOGDIR, base + '.long_run.txt')
    cmd = [sys.executable, os.path.join(BASE, 'ai_player_Q.py'), '--load', q, '--games', '1000', '--history', hist, '--testmode']
    print('Running:', ' '.join(shlex.quote(c) for c in cmd))
    with open(outlog, 'w', encoding='utf-8') as f:
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
            print('Return code', proc.returncode)
        except Exception as e:
            print('Run failed for', q, e)
            continue
    # parse history into per-episode reward sums
    rewards = []
    if not os.path.exists(hist):
        print('History missing after run:', hist)
        continue
    cur = 0.0
    with open(hist, newline='', encoding='utf-8', errors='replace') as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                rew = float(row.get('reward', 0) or 0)
            except:
                rew = 0.0
            cur += rew
            res = (row.get('result') or '').strip().lower()
            if res and res not in ['', 'unsettled', 'retry']:
                rewards.append(cur)
                cur = 0.0
    if cur != 0.0:
        rewards.append(cur)
    # compute cumulative money assuming INITIAL_MONEY from config
    sys.path.insert(0, BASE)
    try:
        from config import INITIAL_MONEY
    except:
        INITIAL_MONEY = 10000
    money = [INITIAL_MONEY]
    for r in rewards:
        money.append(money[-1] + r)
    histories.append(money)
    finals.append(money[-1])
    print('Run', base, 'episodes', len(rewards), 'final money', money[-1])

# plotting
plt.figure(figsize=(12,6))
maxlen = max(len(m) for m in histories)
arr = np.zeros((len(histories), maxlen))
for i,m in enumerate(histories):
    # pad with last value
    padded = m + [m[-1]]*(maxlen - len(m))
    arr[i,:] = padded
    plt.plot(range(len(padded)), padded, alpha=0.4, label=os.path.basename(qt_files[i]))
# mean
mean_series = np.mean(arr, axis=0)
plt.plot(range(len(mean_series)), mean_series, color='k', linewidth=2.5, label='mean')
plt.xlabel('Episode')
plt.ylabel('Money')
plt.title('Top1 group multi-seed money progression (1000 games each)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FIG)
print('Saved figure to', OUT_FIG)

# print summary
for q, f in zip(qt_files, finals):
    print('Q:', os.path.basename(q), 'final_money:', f)
print('Mean final money across seeds:', float(np.mean(finals)))
