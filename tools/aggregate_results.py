import csv
import glob
import os
from statistics import mean


def parse_history(path):
    # ゲーム単位で累積報酬を集計する新ロジック
    game_rewards = []  # 各ゲームの累積報酬
    ep_reward = 0.0    # 現在進行中ゲームの累積報酬
    wins = 0
    losses = 0
    busts = 0
    surrenders = 0
    draws = 0
    games = 0
    terminal_statuses = {'win', 'lose', 'bust', 'surrendered', 'draw'}

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                # row: score,hand_length,action,status,reward
                status = row[3]
                reward = float(row[4])
            except Exception:
                continue

            # ステップ報酬を加算
            ep_reward += reward

            # 終端ならゲームを確定
            if status in terminal_statuses:
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

    # 未終了ゲームは無視（ep_reward リセットされていない残骸）
    avg_reward = mean(game_rewards) if game_rewards else 0.0
    win_rate = wins / games if games else 0.0
    return {
        'path': path,
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'games': games,
        'wins': wins,
        'losses': losses,
        'busts': busts,
        'surrenders': surrenders,
        'draws': draws,
    }


def main():
    files = glob.glob(os.path.join('logs', 'hparam_*.history.csv'))
    results = []
    for fp in files:
        results.append(parse_history(fp))
    results.sort(key=lambda x: (x['avg_reward'], x['win_rate']), reverse=True)
    out_csv = os.path.join('logs', 'summary.csv')
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'avg_reward', 'win_rate', 'games', 'wins', 'losses', 'busts', 'surrenders', 'draws', 'history_path'])
        for i, r in enumerate(results, 1):
            w.writerow([i, r['avg_reward'], r['win_rate'], r['games'], r['wins'], r['losses'], r['busts'], r['surrenders'], r['draws'], r['path']])
    print(f'Wrote summary to {out_csv}. Top 5:')
    for r in results[:5]:
        print(f"avg={r['avg_reward']:.3f} win={r['win_rate']:.3f} games={r['games']} path={r['path']}")


if __name__ == '__main__':
    main()
