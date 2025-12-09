import csv
import argparse
from pathlib import Path
from statistics import mean, median, pstdev

TERMINAL = {'win','lose','bust','surrendered','draw'}
ACTION_COL = 2
STATUS_COL = 3
REWARD_COL = 4


def parse_history(path: Path):
    games = 0
    wins = losses = busts = surrenders = draws = 0
    game_rewards = []
    ep_reward = 0.0
    action_counts = {}
    retries_per_game = []
    retries_current = 0
    actions_current = 0
    max_retries = 0

    def flush_game(status: str):
        nonlocal games, wins, losses, busts, surrenders, draws, ep_reward, retries_current, max_retries
        games += 1
        if status == 'win':
            wins += 1
        elif status == 'lose':
            losses += 1
        elif status == 'bust':
            busts += 1
        elif status == 'surrendered':
            surrenders += 1
        elif status == 'draw':
            draws += 1
        game_rewards.append(ep_reward)
        retries_per_game.append(retries_current)
        if retries_current > max_retries:
            max_retries = retries_current

    with path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                action = row[ACTION_COL].strip()
                status = row[STATUS_COL].strip()
                reward = float(row[REWARD_COL])
            except Exception:
                continue
            ep_reward += reward
            actions_current += 1
            action_counts[action] = action_counts.get(action, 0) + 1
            if action == 'RETRY':
                retries_current += 1
            if status in TERMINAL:
                flush_game(status)
                # reset episode accumulators
                ep_reward = 0.0
                retries_current = 0
                actions_current = 0

    avg_reward = mean(game_rewards) if game_rewards else 0.0
    med_reward = median(game_rewards) if game_rewards else 0.0
    std_reward = pstdev(game_rewards) if len(game_rewards) > 1 else 0.0
    retry_total = sum(retries_per_game)
    retry_any_games = sum(1 for r in retries_per_game if r > 0)
    return {
        'run': path.name.replace('hparam_','').replace('.history.csv',''),
        'games': games,
        'win_rate': wins / games if games else 0.0,
        'avg_reward': avg_reward,
        'median_reward': med_reward,
        'std_reward': std_reward,
        'wins': wins,
        'losses': losses,
        'busts': busts,
        'surrenders': surrenders,
        'draws': draws,
        'retry_total': retry_total,
        'retry_per_game': retry_total / games if games else 0.0,
        'retry_any_rate': retry_any_games / games if games else 0.0,
        'max_retries_game': max_retries,
        'action_counts': action_counts,
    }


def format_action_distribution(action_counts: dict, games: int):
    # return compact distribution string: ACTION=rate%
    total_actions = sum(action_counts.values()) or 1
    parts = []
    for a in sorted(action_counts.keys()):
        cnt = action_counts[a]
        parts.append(f"{a}:{cnt/total_actions:.2f}")
    return ' '.join(parts)


def main():
    ap = argparse.ArgumentParser(description='Compare runs with advanced metrics')
    ap.add_argument('--histories', nargs='+', required=True, help='List of history CSV paths')
    ap.add_argument('--out', default='logs/compare_runs.csv')
    args = ap.parse_args()

    rows = []
    for hp in args.histories:
        p = Path(hp)
        if not p.exists():
            print(f'[WARN] Missing {p}, skipping.')
            continue
        rows.append(parse_history(p))
    if not rows:
        print('[INFO] No valid histories.')
        return

    # write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'run','games','win_rate','avg_reward','median_reward','std_reward','wins','losses','busts','surrenders','draws',
            'retry_total','retry_per_game','retry_any_rate','max_retries_game','action_dist'
        ])
        for r in rows:
            w.writerow([
                r['run'], r['games'], f"{r['win_rate']:.4f}", f"{r['avg_reward']:.4f}", f"{r['median_reward']:.4f}", f"{r['std_reward']:.4f}",
                r['wins'], r['losses'], r['busts'], r['surrenders'], r['draws'], r['retry_total'], f"{r['retry_per_game']:.3f}",
                f"{r['retry_any_rate']:.3f}", r['max_retries_game'], format_action_distribution(r['action_counts'], r['games'])
            ])
    print(f'[INFO] Wrote comparison to {out_path}')
    # print quick ranking by avg_reward desc
    rank = sorted(rows, key=lambda x: x['avg_reward'], reverse=True)
    print('[TOP by avg_reward]')
    for r in rank[:5]:
        print(f"{r['run']} avg={r['avg_reward']:.3f} win={r['win_rate']:.3f} retry_pg={r['retry_per_game']:.2f}")


if __name__ == '__main__':
    main()
