import itertools, os, subprocess, sys, concurrent.futures, time

# recreate expected param grid and run_name helper
def make_run_name(params):
    parts = []
    for k in ['alpha', 'gamma', 'eps_start', 'eps_end', 'eps_decay_type', 'seed']:
        parts.append(f"{k}={params.get(k)}")
    name = "--".join(parts)
    return name.replace(' ', '').replace('.', '_').replace('/', '_').replace('=', '-')

alphas = [0.01, 0.03, 0.05, 0.1]
gammas = [0.9, 0.95, 0.98]
eps_starts = [1.0, 0.5]
eps_ends = [0.01, 0.05]
decay_types = ['const', 'linear', 'exp']
seeds = [0]

grid = []
for a,g,es,ee,dt,s in itertools.product(alphas,gammas,eps_starts,eps_ends,decay_types,seeds):
    grid.append({'alpha':a,'gamma':g,'eps_start':es,'eps_end':ee,'eps_decay_type':dt,'seed':s,'eps_decay_episodes':1000})

qt_dir = os.path.join('logs','qtables')
if not os.path.isdir(qt_dir):
    os.makedirs(qt_dir, exist_ok=True)

existing = set()
for fn in os.listdir(qt_dir):
    if fn.endswith('.pkl'):
        existing.add(os.path.splitext(fn)[0])

# collect missing param dicts
missing = []
for params in grid:
    name = make_run_name(params)
    if name not in existing:
        missing.append((name, params))

print(f'Will run {len(missing)} missing runs (max workers: 8)')

# runner for a single missing run
def run_one(item):
    run_name, params = item
    out_dir = 'logs'
    history_path = os.path.join(out_dir, run_name + '.history.csv')
    out_path = os.path.join(out_dir, run_name + '.txt')
    save_path = os.path.join('logs','qtables', run_name + '.pkl')

    cmd = [sys.executable, 'ai_player_Q.py',
           '--games', str(1000),
           '--history', history_path,
           '--alpha', str(params['alpha']),
           '--gamma', str(params['gamma']),
           '--eps_start', str(params['eps_start']),
           '--eps_end', str(params['eps_end']),
           '--eps_decay_episodes', str(params.get('eps_decay_episodes',1000)),
           '--eps_decay_type', params['eps_decay_type'],
           '--save', save_path]

    print('Starting', run_name)
    start = time.time()
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=60*60*6)
        elapsed = time.time() - start
        print(f'Finished {run_name} in {elapsed:.1f}s, returncode={proc.returncode}')
        return (run_name, True, proc.returncode)
    except subprocess.TimeoutExpired:
        print(f'Timeout for {run_name}')
        return (run_name, False, 'timeout')
    except Exception as e:
        print(f'Error running {run_name}:', e)
        return (run_name, False, str(e))

# run missing tasks with ThreadPoolExecutor
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(run_one, item) for item in missing]
    for fut in concurrent.futures.as_completed(futures):
        try:
            r = fut.result()
            print('Result:', r)
            results.append(r)
        except Exception as e:
            print('Future exception:', e)

print('All tasks submitted. Summary:')
for r in results:
    print(r)
