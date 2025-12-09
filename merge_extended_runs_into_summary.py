import csv
from pathlib import Path

def load_extended():
    ext_path = Path('logs/extended_runs/extended_runs_summary.csv')
    rows = []
    if not ext_path.exists():
        return rows
    with ext_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def compute_basic_row(ext_row):
    # Reconstruct components for summary format
    # counts for actions not directly needed
    games = int(float(ext_row['games']))
    # We lack direct win, lose, draw counts in extended_runs_summary -> need to recompute? Not stored.
    # Simplify: we cannot recompute without original counts; adjust aggregator to include them would be ideal.
    # Fallback: skip merge if missing essential fields.
    needed = ['win_rate','avg_reward']
    for n in needed:
        if n not in ext_row:
            return None
    # Provide placeholder zeros for columns we cannot reconstruct here.
    return {
        'rank':'',
        'avg_reward': ext_row['avg_reward'],
        'win_rate': ext_row['win_rate'],
        'games': games,
        'wins':'',
        'losses':'',
        'busts':'',
        'surrenders':'',
        'draws':'',
        'history_path': f"logs/extended_runs/{ext_row['run']}/history.csv",
    }

def main():
    summary_path = Path('logs/summary.csv')
    if not summary_path.exists():
        print('No summary.csv to merge into.')
        return
    with summary_path.open('r', encoding='utf-8') as f:
        existing = list(csv.DictReader(f))
    max_rank = 0
    for r in existing:
        try:
            max_rank = max(max_rank, int(r['rank']))
        except Exception:
            pass
    extended_rows = load_extended()
    new_rows_basic = []
    for er in extended_rows:
        br = compute_basic_row(er)
        if br:
            new_rows_basic.append(br)
    # Assign ranks after current max rank based on avg_reward descending (higher better)
    def parse_avg(v):
        try:
            return float(v)
        except Exception:
            return -1e9
    new_rows_basic.sort(key=lambda r: parse_avg(r['avg_reward']), reverse=True)
    next_rank = max_rank + 1
    for r in new_rows_basic:
        r['rank'] = str(next_rank)
        next_rank += 1
    # Write merged file
    out_path = Path('logs/summary_with_extended.csv')
    fieldnames = ['rank','avg_reward','win_rate','games','wins','losses','busts','surrenders','draws','history_path']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k,'') for k in fieldnames})
        for r in new_rows_basic:
            w.writerow({k: r.get(k,'') for k in fieldnames})
    print(f'Wrote {out_path} with {len(existing)} existing + {len(new_rows_basic)} extended rows.')

if __name__ == '__main__':
    main()
