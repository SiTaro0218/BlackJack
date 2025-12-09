import argparse
import csv
import os
from datetime import datetime

# Reuse aggregation from extended runs
try:
    from extended_runs_aggregate import aggregate_history
except Exception:
    aggregate_history = None


def latest_sweep_dir(base):
    if not os.path.isdir(base):
        return None
    stamps = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not stamps:
        return None
    stamps.sort()
    return os.path.join(base, stamps[-1])


def summarize(sweep_path):
    rows = []
    for name in os.listdir(sweep_path):
        run_dir = os.path.join(sweep_path, name)
        if not os.path.isdir(run_dir):
            continue
        hist = os.path.join(run_dir, 'history.csv')
        if not os.path.exists(hist):
            continue
        if aggregate_history is None:
            # Fallback minimal metrics if import failed
            with open(hist, 'r', encoding='utf-8') as f:
                lines = f.read().strip().splitlines()
            games = sum(1 for ln in lines if ',win,' in ln or ',lose,' in ln or ',draw,' in ln or ',bust,' in ln or ',surrendered,' in ln)
            rows.append({'run': name, 'games': games, 'avg_reward': 0.0, 'total_reward': 0.0})
            continue
        m = aggregate_history(hist)
        rows.append({'run': name, **m})

    # Sort by avg_reward (higher is better)
    rows.sort(key=lambda r: r.get('avg_reward', 0.0), reverse=True)

    out_csv = os.path.join(sweep_path, 'summary.csv')
    fieldnames = ['run','games','total_reward','avg_reward','avg_reward_ci95_low','avg_reward_ci95_high','net_profit_per_100','total_retries','avg_retries','retry_rate','win_rate','lose_rate','draw_rate','surrender_rate','bust_rate','hit_count','stand_count','double_down_count','surrender_count','retry_count','penalty_total','win_reward_total','lose_reward_total','draw_reward_total','surrender_reward_total','bust_reward_total']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote {out_csv} ({len(rows)} runs)')

    # Also write a simple top10 text view
    top_txt = os.path.join(sweep_path, 'top10.txt')
    with open(top_txt, 'w', encoding='utf-8') as f:
        for i, r in enumerate(rows[:10]):
            f.write(f"{i+1:02d}. {r['run']}  avg={r.get('avg_reward',0):.4f}  ci95=({r.get('avg_reward_ci95_low',0):.4f},{r.get('avg_reward_ci95_high',0):.4f})  per100={r.get('net_profit_per_100',0):.2f}\n")
    print(f'Wrote {top_txt}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-path', type=str, default='', help='Path to logs/sweeps/<stamp>. If empty, picks the latest.')
    args = parser.parse_args()

    base = os.path.join('logs','sweeps')
    sweep_path = args.sweep_path or latest_sweep_dir(base)
    if not sweep_path:
        print('No sweep directory found.')
        return
    summarize(sweep_path)


if __name__ == '__main__':
    main()
