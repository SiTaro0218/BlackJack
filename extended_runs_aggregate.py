import csv
import os
from math import sqrt

def aggregate_history(history_path):
    total_reward = 0.0
    games = 0
    hit = stand = dd = surrender = retry = 0
    wins = loses = draws = surrenders = busts = 0
    retries_total = 0
    penalty_total = 0.0
    win_reward_total = 0.0
    lose_reward_total = 0.0
    draw_reward_total = 0.0
    surrender_reward_total = 0.0
    bust_reward_total = 0.0
    rewards_per_game = []
    game_reward = 0.0

    with open(history_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get('status','') or '').lower()
            action = row.get('action','')
            reward_str = row.get('reward','0')
            reward = float(reward_str or 0)
            if action == 'HIT': hit += 1
            elif action == 'STAND': stand += 1
            elif action == 'DOUBLE_DOWN': dd += 1
            elif action == 'SURRENDER': surrender += 1
            elif action == 'RETRY':
                retry += 1
                retries_total += 1
                if reward < 0:
                    penalty_total += reward
            game_reward += reward
            if status in ('win','lose','draw','surrendered','bust'):
                games += 1
                total_reward += game_reward
                rewards_per_game.append(game_reward)
                if status == 'WIN':
                    wins += 1
                    win_reward_total += game_reward
                elif status == 'LOSE':
                    loses += 1
                    lose_reward_total += game_reward
                elif status == 'DRAW':
                    draws += 1
                    draw_reward_total += game_reward
                elif status == 'SURRENDER':
                    surrenders += 1
                    surrender_reward_total += game_reward
                elif status == 'BUST':
                    busts += 1
                    bust_reward_total += game_reward
                game_reward = 0.0

    avg_reward = total_reward / games if games else 0.0
    if rewards_per_game:
        mean = avg_reward
        var = sum((r-mean)**2 for r in rewards_per_game) / max(1, (len(rewards_per_game)-1))
        se = sqrt(var / len(rewards_per_game))
        ci95_low = mean - 1.96*se
        ci95_high = mean + 1.96*se
    else:
        ci95_low = ci95_high = avg_reward

    avg_retries = retries_total / games if games else 0.0
    total_actions = hit+stand+dd+surrender+retry
    retry_rate = retry / total_actions if total_actions else 0.0
    return {
        'games': games,
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'avg_reward_ci95_low': ci95_low,
        'avg_reward_ci95_high': ci95_high,
        'total_retries': retries_total,
        'avg_retries': avg_retries,
        'retry_rate': retry_rate,
        'win_rate': wins/games if games else 0.0,
        'lose_rate': loses/games if games else 0.0,
        'draw_rate': draws/games if games else 0.0,
        'surrender_rate': surrenders/games if games else 0.0,
        'bust_rate': busts/games if games else 0.0,
        'hit_count': hit,
        'stand_count': stand,
        'double_down_count': dd,
        'surrender_count': surrender,
        'retry_count': retry,
        'penalty_total': penalty_total,
        'win_reward_total': win_reward_total,
        'lose_reward_total': lose_reward_total,
        'draw_reward_total': draw_reward_total,
        'surrender_reward_total': surrender_reward_total,
        'bust_reward_total': bust_reward_total,
        'net_profit_per_100': (avg_reward*100.0),
    }

def main():
    base = os.path.join('logs','extended_runs')
    rows = []
    for d in os.listdir(base):
        run_dir = os.path.join(base, d)
        if not os.path.isdir(run_dir):
            continue
        hist = os.path.join(run_dir, 'history.csv')
        if not os.path.exists(hist):
            continue
        m = aggregate_history(hist)
        desc = ''
        if '_10k' in d:
            if 'ps05' in d: desc = 'alpha=0.15 gamma=0.95 penalty_scale=0.5'
            elif 'ps03' in d: desc = 'alpha=0.15 gamma=0.95 penalty_scale=0.3'
            elif 'ps07' in d: desc = 'alpha=0.15 gamma=0.95 penalty_scale=0.7'
        rows.append({ 'run': d, 'desc': desc, **m })
    out = os.path.join(base, 'extended_runs_summary.csv')
    fieldnames = ['run','desc','games','total_reward','avg_reward','avg_reward_ci95_low','avg_reward_ci95_high','net_profit_per_100','total_retries','avg_retries','retry_rate','win_rate','lose_rate','draw_rate','surrender_rate','bust_rate','hit_count','stand_count','double_down_count','surrender_count','retry_count','penalty_total','win_reward_total','lose_reward_total','draw_reward_total','surrender_reward_total','bust_reward_total']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()
