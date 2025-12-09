import csv
import glob
import os
from statistics import mean


def parse_history(path):
    game_rewards = []
    ep_reward = 0.0
    wins = losses = busts = surrenders = draws = games = 0
    terminal = {'win','lose','bust','surrendered','draw'}
    with open(path,'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # header skip
        for row in reader:
            if len(row) < 5:
                continue
            status = row[3]
            try:
                reward = float(row[4])
            except Exception:
                continue
            ep_reward += reward
            if status in terminal:
                game_rewards.append(ep_reward)
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
                ep_reward = 0.0
    return {
        'path': path,
        'avg_reward': mean(game_rewards) if game_rewards else 0.0,
        'win_rate': wins / games if games else 0.0,
        'games': games,
        'wins': wins,
        'losses': losses,
        'busts': busts,
        'surrenders': surrenders,
        'draws': draws,
    }


def main():
    files = glob.glob(os.path.join('logs','hparam_*_rerun.history.csv'))
    if not files:
        print('[INFO] No rerun histories found.')
        return
    results = [parse_history(fp) for fp in files]
    results.sort(key=lambda x: (x['avg_reward'], x['win_rate']), reverse=True)
    out_csv = os.path.join('logs','summary_rerun.csv')
    with open(out_csv,'w',encoding='utf-8',newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank','avg_reward','win_rate','games','wins','losses','busts','surrenders','draws','history_path'])
        for i,r in enumerate(results,1):
            w.writerow([i,r['avg_reward'],r['win_rate'],r['games'],r['wins'],r['losses'],r['busts'],r['surrenders'],r['draws'],r['path']])
    print(f'[INFO] Wrote rerun summary to {out_csv}. Top:')
    for r in results[:5]:
        print(f"avg={r['avg_reward']:.3f} win={r['win_rate']:.3f} games={r['games']} path={r['path']}")

if __name__ == '__main__':
    main()
