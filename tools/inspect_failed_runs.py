"""
Inspect alpha=0.2 experiment results for failed / early-terminated runs.
- Reads `results_alpha0_2.csv` and `results_alpha0_2_long.csv` if present.
- Flags runs with episodes < 500 or episodes == 0 or mov_avg_100 == 0.
- For each flagged run prints metadata and last ~200 lines of its logfile (with encoding fallbacks).
- Writes output to `tools/failed_runs_report.txt`.
Run: python .\tools\inspect_failed_runs.py
"""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parent.parent
CSV_FILES = [ROOT / 'results_alpha0_2.csv', ROOT / 'results_alpha0_2_long.csv']
OUT = ROOT / 'tools' / 'failed_runs_report.txt'
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in CSV_FILES:
    if not f.exists():
        continue
    with f.open('r', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # normalize episodes
            try:
                r['episodes'] = int(r.get('episodes', '') or 0)
            except ValueError:
                r['episodes'] = 0
            try:
                r['mov_avg_100'] = float(r.get('mov_avg_100', '') or 0.0)
            except ValueError:
                r['mov_avg_100'] = 0.0
            rows.append(r)

# criteria: episodes < 500 OR episodes == 0 OR mov_avg_100 == 0.0
sus = [r for r in rows if (r['episodes'] < 500 or r['episodes'] == 0 or r['mov_avg_100'] == 0.0)]
# sort by episodes ascending
sus = sorted(sus, key=lambda x: x['episodes'])

def read_tail(path, max_lines=200):
    p = Path(path)
    if not p.exists():
        return f"<logfile not found: {path}>"
    # try encodings
    encs = ['utf-8', 'cp932', 'latin-1']
    text = None
    for e in encs:
        try:
            text = p.read_text(encoding=e)
            break
        except Exception:
            continue
    if text is None:
        # fallback binary
        try:
            data = p.read_bytes()
            text = data.decode('utf-8', errors='replace')
        except Exception as ex:
            return f"<failed to read logfile {path}: {ex}>"
    lines = text.splitlines()
    tail = lines[-max_lines:]
    return '\n'.join(tail)

with OUT.open('w', encoding='utf-8') as out:
    out.write('Failed / short runs report\n')
    out.write('Source CSV files: ' + ', '.join(str(p) for p in CSV_FILES if p.exists()) + '\n\n')
    if not sus:
        out.write('No suspicious runs found by criteria (episodes < 500 or mov_avg_100 == 0)\n')
        print('No suspicious runs found; report created.')
        sys.exit(0)

    out.write(f'Found {len(sus)} suspicious runs (sorted by episodes asc)\n\n')
    for i, r in enumerate(sus, 1):
        out.write(f"=== Run {i} / {len(sus)} ===\n")
        out.write(f"run_name: {r.get('run_name')}\n")
        out.write(f"alpha: {r.get('alpha')}  gamma: {r.get('gamma')}  eps_start: {r.get('eps_start')}  eps_end: {r.get('eps_end')}  eps_decay_type: {r.get('eps_decay_type')}  seed: {r.get('seed')}\n")
        out.write(f"episodes: {r.get('episodes')}  avg_reward: {r.get('avg_reward')}  mov_avg_100: {r.get('mov_avg_100')}\n")
        logfile = r.get('logfile') or ''
        # logfile may be relative; resolve from ROOT
        lf_path = (ROOT / logfile) if logfile else (ROOT / 'logs' / (r.get('run_name') + '.txt'))
        out.write(f"logfile: {lf_path}\n")
        out.write('\n--- logfile tail (last 200 lines) ---\n')
        tail = read_tail(lf_path, max_lines=200)
        out.write(tail + '\n')
        out.write('\n\n')

print(f'Report written to {OUT}')
