import os, subprocess, shlex, sys
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
qpath = os.path.join(BASE, 'logs', 'qtables', 'alpha-0_1--gamma-0_9--eps_start-1_0--eps_end-0_05--eps_decay_type-const--seed-0.pkl')
qpath = os.path.normpath(qpath)
print('Using qtable:', qpath)
out_log = os.path.join(BASE, 'logs', 'test_top1_long_run.txt')
history = os.path.join(BASE, 'logs', 'test_top1_long.history.csv')
# run 1000 games by default; can be overridden by first CLI arg
games = 1000
if len(sys.argv) > 1:
    try:
        games = int(sys.argv[1])
    except:
        pass
cmd = [sys.executable, os.path.join(BASE, 'ai_player_Q.py'), '--load', qpath, '--games', str(games), '--history', history, '--testmode']
print('Command:', ' '.join(shlex.quote(c) for c in cmd))
with open(out_log, 'w', encoding='utf-8') as f:
    try:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        print('Return code:', proc.returncode)
    except subprocess.TimeoutExpired:
        print('Run timed out')
    except Exception as e:
        print('Run failed:', e)
print('Wrote log to', out_log)
