import os, time, subprocess, shlex, sys
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
missing_file = os.path.join(BASE, 'logs', 'missing_runs_seed1-4.txt')
if not os.path.exists(missing_file):
    print('Missing list not found:', missing_file); sys.exit(2)
with open(missing_file, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]
if not lines:
    print('No missing runs'); sys.exit(0)
# pick first missing
run_name = lines[0]
print('Selected run:', run_name)
# parse to params if needed
qt_path = os.path.join(BASE, 'logs', 'qtables', run_name + '.pkl')
history = os.path.join(BASE, 'logs', run_name + '.smoke.history.csv')
out_log = os.path.join(BASE, 'logs', run_name + '.smoke.txt')
cmd = [sys.executable, os.path.join(BASE, 'ai_player_Q.py'), '--games', '100', '--history', history, '--save', qt_path]
print('Command:', ' '.join(shlex.quote(c) for c in cmd))
start = time.time()
with open(out_log, 'w', encoding='utf-8') as f:
    try:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    except Exception as e:
        print('Run failed:', e); sys.exit(1)
elapsed = time.time() - start
print('Elapsed seconds:', round(elapsed,2))
print('Wrote log to', out_log)
print('Qtable saved to (if successful):', qt_path)
