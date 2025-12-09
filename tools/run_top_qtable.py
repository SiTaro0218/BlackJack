import os, subprocess, shlex, sys
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
qpath = os.path.join(BASE, 'logs', 'qtables', 'alpha-0_1--gamma-0_9--eps_start-1_0--eps_end-0_05--eps_decay_type-const--seed-0.pkl')
qpath = os.path.normpath(qpath)
print('Using qtable:', qpath)
out_log = os.path.join(BASE, 'logs', 'test_top1_run.txt')
history = os.path.join(BASE, 'logs', 'test_top1.history.csv')
cmd = [sys.executable, os.path.join(BASE, 'ai_player_Q.py'), '--load', qpath, '--games', '100', '--history', history, '--testmode']
print('Command:', ' '.join(shlex.quote(c) for c in cmd))
with open(out_log, 'w', encoding='utf-8') as f:
    try:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=300)
        print('Return code:', proc.returncode)
    except Exception as e:
        print('Run failed:', e)
print('Wrote log to', out_log)
