import os, sys, subprocess, shlex
import concurrent.futures
import time
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
missing_file = os.path.join(BASE, 'logs', 'missing_runs_seed1-4.txt')
if not os.path.exists(missing_file):
    print('Missing file not found:', missing_file); sys.exit(2)
with open(missing_file, 'r', encoding='utf-8') as f:
    runs = [l.strip() for l in f if l.strip()]
if not runs:
    print('No missing runs to execute'); sys.exit(0)

OUT_LOG_DIR = os.path.join(BASE, 'logs')
QT_DIR = os.path.join(BASE, 'logs', 'qtables')
os.makedirs(QT_DIR, exist_ok=True)

PY = sys.executable
SCRIPT = os.path.join(BASE, 'ai_player_Q.py')
GAMES = 1000
TIMEOUT = 3600

def run_one(rn):
    qt = os.path.join(QT_DIR, rn + '.pkl')
    hist = os.path.join(OUT_LOG_DIR, rn + '.long.history.csv')
    out = os.path.join(OUT_LOG_DIR, rn + '.long.txt')
    cmd = [PY, SCRIPT, '--games', str(GAMES), '--history', hist, '--save', qt]
    start = time.time()
    try:
        with open(out, 'w', encoding='utf-8') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT)
        elapsed = time.time() - start
        return (rn, True, proc.returncode, elapsed, out, qt)
    except subprocess.TimeoutExpired:
        return (rn, False, 'timeout', TIMEOUT, out, qt)
    except Exception as e:
        return (rn, False, str(e), 0, out, qt)

print('Executing', len(runs), 'missing runs with workers=8')
results = []
start_all = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(run_one, rn): rn for rn in runs}
    for fut in concurrent.futures.as_completed(futures):
        rn = futures[fut]
        try:
            res = fut.result()
            results.append(res)
            print('Done:', res[0], 'ok=', res[1], 'ret=', res[2], 'time(s)=', round(res[3],2))
        except Exception as e:
            print('Exception for', rn, e)
end_all = time.time()
print('All tasks finished. Total wall time(s):', round(end_all - start_all,2))
# write summary
with open(os.path.join(BASE, 'logs', 'missing_runs_summary.txt'), 'w', encoding='utf-8') as f:
    for r in results:
        f.write(','.join(map(str, r)) + '\n')
print('Wrote logs/missing_runs_summary.txt')
