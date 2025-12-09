import sys
import csv
from pathlib import Path

def summarize(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    wins = draws = losses = 0
    games = 0
    rewards = []  # per-game cumulative rewards (includes RETRY penalties)
    ep_reward = 0.0  # accumulate rewards until a terminal result
    with p.open(newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            if row[0].startswith('score'):
                # skip header (some files have a corrupted long header)
                continue
            try:
                result = row[-2].strip()
                reward = float(row[-1])
            except Exception:
                continue
            # accumulate step reward (includes 0 for non-terminal steps and -penalty for RETRY)
            ep_reward += reward
            # when terminal, record game outcome and push cumulative reward, then reset accumulator
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
    avg = (sum(rewards) / games) if games else 0.0
    win_rate = (wins / games) if games else 0.0
    print(f"file={p}")
    print(f"games={games}")
    print(f"win_rate={win_rate:.3f}")
    print(f"avg_reward={avg:.3f}")
    print(f"win={wins}, draw={draws}, lose={losses}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/summarize_history.py <history_csv>')
        sys.exit(1)
    summarize(sys.argv[1])
