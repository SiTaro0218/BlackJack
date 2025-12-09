import csv
from pathlib import Path

csv_path = Path(__file__).resolve().parents[1] / 'results.csv'
rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # convert fields
        try:
            r['episodes'] = int(r.get('episodes', '0') or 0)
        except:
            r['episodes'] = 0
        try:
            r['avg_reward'] = float(r.get('avg_reward', '0') or 0)
        except:
            r['avg_reward'] = 0.0
        try:
            r['mov_avg_100'] = float(r.get('mov_avg_100', '0') or 0)
        except:
            r['mov_avg_100'] = 0.0
        rows.append(r)

# filter completed runs (episodes>0)
completed = [r for r in rows if r['episodes'] > 0]
if not completed:
    print('No completed runs found in results.csv')
    raise SystemExit(0)

# top by mov_avg_100 (higher is better)
top_mov = sorted(completed, key=lambda x: x['mov_avg_100'], reverse=True)[:5]
# top by avg_reward
top_avg = sorted(completed, key=lambda x: x['avg_reward'], reverse=True)[:5]

print('Top 5 by mov_avg_100:')
for r in top_mov:
    run = r['run_name']
    print(f"{run}  seed={r['seed']} episodes={r['episodes']} mov_avg_100={r['mov_avg_100']} avg_reward={r['avg_reward']}")

print('\nTop 5 by avg_reward:')
for r in top_avg:
    run = r['run_name']
    print(f"{run}  seed={r['seed']} episodes={r['episodes']} avg_reward={r['avg_reward']} mov_avg_100={r['mov_avg_100']}")

# best per seed (by mov_avg_100)
print('\nBest per seed (by mov_avg_100):')
seeds = sorted(set(int(r['seed']) for r in completed))
for s in seeds:
    rs = [r for r in completed if int(r['seed']) == s]
    best = max(rs, key=lambda x: x['mov_avg_100'])
    print(f"seed={s}: {best['run_name']}  episodes={best['episodes']} mov_avg_100={best['mov_avg_100']} avg_reward={best['avg_reward']}")

# Suggest qtable/history paths for the top overall run
best_overall = top_mov[0]
run_name = best_overall['run_name']
base = Path(__file__).resolve().parents[1]
qtable = base / 'logs' / 'qtables' / (run_name + '.pkl')
history = base / 'logs' / (run_name + '.history.csv')
print('\nBest overall run (by mov_avg_100):')
print(run_name)
print('qtable expected at:', qtable)
print('history expected at:', history)

# also consider only sufficiently long runs to avoid tiny-run bias
long_runs = [r for r in completed if r['episodes'] >= 900]
if long_runs:
    best_long = max(long_runs, key=lambda x: x['mov_avg_100'])
    print('\nBest among runs with episodes>=900:')
    print(f"{best_long['run_name']}  seed={best_long['seed']} episodes={best_long['episodes']} mov_avg_100={best_long['mov_avg_100']} avg_reward={best_long['avg_reward']}")
    bl_q = base / 'logs' / 'qtables' / (best_long['run_name'] + '.pkl')
    bl_h = base / 'logs' / (best_long['run_name'] + '.history.csv')
    print('qtable expected at:', bl_q)
    print('history expected at:', bl_h)
else:
    print('\nNo runs with episodes>=900 found to compare for long-run best.')

# best per seed among long runs
if long_runs:
    print('\nBest per seed among runs with episodes>=900:')
    long_seeds = sorted(set(int(r['seed']) for r in long_runs))
    for s in long_seeds:
        rs = [r for r in long_runs if int(r['seed']) == s]
        if not rs:
            continue
        bests = max(rs, key=lambda x: x['mov_avg_100'])
        print(f"seed={s}: {bests['run_name']}  episodes={bests['episodes']} mov_avg_100={bests['mov_avg_100']} avg_reward={bests['avg_reward']}")
