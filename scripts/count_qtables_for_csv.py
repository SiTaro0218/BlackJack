import csv
import glob
import os
import sys

def float_to_token(x):
    s = str(x)
    return s.replace('.', '_')

if len(sys.argv) < 2:
    print('Usage: python count_qtables_for_csv.py path/to/csv [qtables_dir]')
    sys.exit(2)

csvpath = sys.argv[1]
qtables_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('logs','autocollect_FULL_20251110_121110','qtables')

if not os.path.exists(csvpath):
    print('CSV not found', csvpath)
    sys.exit(2)

rows = []
with open(csvpath,'r',newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        alpha = r['alpha']
        gamma = r['gamma']
        bins = r['bins']
        eps_decay_type = r['eps_decay_type']
        eps_start = r['eps_start']
        eps_end = r['eps_end']
        a = float_to_token(alpha)
        g = float_to_token(gamma)
        es = float_to_token(eps_start)
        ee = float_to_token(eps_end)
        pattern = os.path.join(qtables_dir, f"*alpha-{a}--gamma-{g}--eps_start-{es}--eps_end-{ee}--eps_decay_type-{eps_decay_type}--bins-{bins}--seed-*.pkl")
        matches = glob.glob(pattern)
        rows.append({'alpha':alpha,'gamma':gamma,'bins':bins,'eps_decay_type':eps_decay_type,'eps_start':eps_start,'eps_end':eps_end,'found':len(matches)})

for r in rows:
    print(r)
print('Total qtables:', sum([r['found'] for r in rows]))
