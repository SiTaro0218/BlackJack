import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

THIS_PY = Path(__file__).resolve()
ROOT = THIS_PY.parent.parent


def summarize_history(path: Path):
    wins = draws = losses = 0
    games = 0
    rewards = []  # per-game cumulative rewards (includes RETRY penalties)
    ep_reward = 0.0
    with path.open(newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            if row[0].startswith('score'):
                continue
            try:
                result = row[-2].strip()
                reward = float(row[-1])
            except Exception:
                continue
            # accumulate step reward (0 for continue, negative for RETRY, final payout on terminal)
            ep_reward += reward
            if result in ('win','lose','draw','bust'):
                if result == 'win':
                    wins += 1
                elif result == 'draw':
                    draws += 1
                else:
                    losses += 1
                games += 1
                rewards.append(ep_reward)
                ep_reward = 0.0
    avg_reward = (sum(rewards) / games) if games else 0.0
    win_rate = (wins / games) if games else 0.0
    return {
        'file': str(path),
        'games': games,
        'win': wins,
        'draw': draws,
        'lose': losses,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
    }


def run_once(python_exe: str, qtable: Path, games: int, out_hist: Path):
    cmd = [
        python_exe,
        str(ROOT / 'ai_player_Q.py'),
        '--games', str(games),
        '--testmode',
        '--load', str(qtable),
        '--history', str(out_hist),
    ]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--load', required=True, help='Path to QTable .pkl')
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--outdir', default=None, help='Output dir for histories and summary')
    args = p.parse_args()

    qtable = Path(args.load).resolve()
    if not qtable.exists():
        print(f'QTable not found: {qtable}')
        sys.exit(1)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(args.outdir) if args.outdir else (ROOT / 'logs' / f'eval_multi_{stamp}')
    outdir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    per_run = []

    for i in range(1, args.repeats + 1):
        hist = outdir / f'run_{i:02d}.history.csv'
        print(f'=== Repeat {i}/{args.repeats}: games={args.games}, history={hist.name} ===')
        rc = run_once(python_exe, qtable, args.games, hist)
        if rc != 0:
            print(f'Run {i} failed with exit code {rc}')
        time.sleep(0.2)  # small stagger
        stats = summarize_history(hist)
        print(f"  -> games={stats['games']}, win_rate={stats['win_rate']:.3f}, avg_reward={stats['avg_reward']:.3f}")
        per_run.append(stats)

    # aggregate
    win_rates = [s['win_rate'] for s in per_run]
    avg_rewards = [s['avg_reward'] for s in per_run]
    pooled_games = sum(s['games'] for s in per_run)
    pooled_wins = sum(s['win'] for s in per_run)
    pooled_draws = sum(s['draw'] for s in per_run)
    pooled_losses = sum(s['lose'] for s in per_run)
    pooled_win_rate = pooled_wins / pooled_games if pooled_games else 0.0
    pooled_avg_reward = sum(s['avg_reward'] * s['games'] for s in per_run) / pooled_games if pooled_games else 0.0

    # naive normal-approx 95% CI for win_rate across pooled trials
    import math
    ci95 = 1.96 * math.sqrt(pooled_win_rate * (1 - pooled_win_rate) / pooled_games) if pooled_games else 0.0

    # write summary CSV
    summary_csv = outdir / 'summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run','games','win','draw','lose','win_rate','avg_reward','history'])
        for idx, s in enumerate(per_run, start=1):
            w.writerow([idx, s['games'], s['win'], s['draw'], s['lose'], f"{s['win_rate']:.6f}", f"{s['avg_reward']:.6f}", s['file']])
        w.writerow([])
        w.writerow(['aggregate_mean', '', '', '', '', f"{mean(win_rates):.6f}", f"{mean(avg_rewards):.6f}", 'means over runs'])
        w.writerow(['aggregate_std', '', '', '', '', f"{pstdev(win_rates):.6f}", f"{pstdev(avg_rewards):.6f}", 'population std'])
        w.writerow(['pooled', pooled_games, pooled_wins, pooled_draws, pooled_losses, f"{pooled_win_rate:.6f}", f"{pooled_avg_reward:.6f}", f"95% CI ±{ci95:.6f}"])

    print('=== Aggregate over runs ===')
    print(f"mean win_rate={mean(win_rates):.3f} (std={pstdev(win_rates):.3f})")
    print(f"mean avg_reward={mean(avg_rewards):.3f} (std={pstdev(avg_rewards):.3f})")
    print(f"pooled win_rate={pooled_win_rate:.3f} ± {ci95:.3f} (n={pooled_games})")
    print(f"pooled avg_reward={pooled_avg_reward:.3f}")
    print(f"Wrote per-run and aggregate summary to {summary_csv}")

if __name__ == '__main__':
    main()
