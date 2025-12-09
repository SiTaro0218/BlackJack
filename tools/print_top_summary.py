import csv
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'summary_table.csv')
path = os.path.normpath(path)

if not os.path.exists(path):
    print('ERROR: figures/summary_table.csv not found at', path)
    sys.exit(2)

rows = []
with open(path, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            row['mov_mean'] = float(row.get('mov_mean', 'nan'))
        except:
            row['mov_mean'] = float('nan')
        try:
            row['avg_mean'] = float(row.get('avg_mean', 'nan'))
        except:
            row['avg_mean'] = float('nan')
        rows.append(row)

if not rows:
    print('No rows found in summary_table.csv')
    sys.exit(0)

rows_sorted_mov = sorted(rows, key=lambda x: x['mov_mean'], reverse=True)
rows_sorted_avg = sorted(rows, key=lambda x: x['avg_mean'], reverse=True)

print('Top 5 by mov_mean:')
for r in rows_sorted_mov[:5]:
    print(r.get('alpha'), r.get('gamma'), r.get('eps_decay_type'), r.get('eps_start'), r.get('eps_end'), 'n_runs', r.get('n_runs'), 'mov_mean', r.get('mov_mean'), 'avg_mean', r.get('avg_mean'))

print('\nTop 5 by avg_mean:')
for r in rows_sorted_avg[:5]:
    print(r.get('alpha'), r.get('gamma'), r.get('eps_decay_type'), r.get('eps_start'), r.get('eps_end'), 'n_runs', r.get('n_runs'), 'avg_mean', r.get('avg_mean'), 'mov_mean', r.get('mov_mean'))
