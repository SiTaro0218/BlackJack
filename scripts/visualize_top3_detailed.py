"""Detailed visualization for top-3 combos.
1) For each qtable in top3 CSV, run greedy evaluation for n_eval_episodes (default 100) to collect per-episode totals.
2) Save per-qtable raw results to figures/top3_detailed_raw_<ts>.csv
3) Create boxplots and per-seed bar plots and save PNGs
4) Render best qtable (highest mean) for render_episodes (default 5) and save GIF
"""
import os
import sys
import glob
import argparse
import datetime
import pickle
import pandas as pd
import numpy as np

TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from enum import Enum
from classes import QTable

class Action(Enum):
    GO_LEFT = 0
    GO_RIGHT = 1
    UNDEFINED = 2

ACTION_ID = { Action.GO_LEFT:0, Action.GO_RIGHT:1 }

def get_state(observation, bins_preset: str = 'original'):
    if bins_preset == 'coarse':
        pos_bins = np.linspace(-4.8, 4.8, 4)
        vel_bins = np.linspace(-3.0, 3.0, 4)
        ang_bins = np.linspace(-0.5, 0.5, 6)
        angvel_bins = np.linspace(-2.0, 2.0, 6)
    elif bins_preset == 'medium':
        pos_bins = np.linspace(-4.8, 4.8, 6)
        vel_bins = np.linspace(-3.0, 3.0, 6)
        ang_bins = np.linspace(-0.5, 0.5, 10)
        angvel_bins = np.linspace(-2.0, 2.0, 10)
    elif bins_preset == 'fine':
        pos_bins = np.linspace(-4.8, 4.8, 10)
        vel_bins = np.linspace(-3.0, 3.0, 10)
        ang_bins = np.linspace(-0.5, 0.5, 20)
        angvel_bins = np.linspace(-2.0, 2.0, 20)
    else:
        pos_bins = np.array([-4.8, -2.4, 0, 2.4, 4.8])
        vel_bins = np.array([-3.0, -1.5, 0, 1.5, 3.0])
        ang_bins = np.array([-0.4, -0.2, 0, 0.2, 0.4])
        angvel_bins = np.array([-2.0, -1.0, 0, 1.0, 2.0])

    cart_pos = np.digitize(observation[0], bins=pos_bins)
    cart_vel = np.digitize(observation[1], bins=vel_bins)
    pole_ang = np.digitize(observation[2], bins=ang_bins)
    pole_vel = np.digitize(observation[3], bins=angvel_bins)

    return (cart_pos, cart_vel, pole_ang, pole_vel)


def find_qtables_for_combo(qtables_dir, alpha, gamma, bins, eps_decay_type, eps_start, eps_end):
    def float_to_token(x):
        return str(x).replace('.', '_')
    a = float_to_token(alpha)
    g = float_to_token(gamma)
    es = float_to_token(eps_start)
    ee = float_to_token(eps_end)
    pattern = os.path.join(qtables_dir, f"*alpha-{a}--gamma-{g}--eps_start-{es}--eps_end-{ee}--eps_decay_type-{eps_decay_type}--bins-{bins}--seed-*.pkl")
    return sorted(glob.glob(pattern))


def eval_qtable(qtable_path, n_episodes=100, max_steps=500):
    with open(qtable_path,'rb') as f:
        loaded = pickle.load(f)
    qtable = QTable(action_class=Action, default_value=0)
    if isinstance(loaded, dict) and 'table' in loaded:
        qtable.table = loaded['table']
        meta = loaded.get('meta', {})
    elif isinstance(loaded, dict):
        qtable.table = loaded
        meta = {}
    else:
        qtable.load(qtable_path); meta = {}
    bins_preset = meta.get('bins','original')
    import gymnasium as gym
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    totals = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs, bins_preset=bins_preset)
        total = 0
        for t in range(max_steps):
            best_action = qtable.get_best_action(state)
            try:
                act_id = ACTION_ID[best_action]
            except Exception:
                act_id = int(best_action)
            obs, reward, done, truncated, info = env.step(act_id)
            state = get_state(obs, bins_preset=bins_preset)
            total += reward
            if done:
                break
        totals.append(total)
    try: env.close()
    except: pass
    return totals


def render_qtable_gif(qtable_path, out_path, n_episodes=5, max_steps=500):
    with open(qtable_path,'rb') as f:
        loaded = pickle.load(f)
    qtable = QTable(action_class=Action, default_value=0)
    if isinstance(loaded, dict) and 'table' in loaded:
        qtable.table = loaded['table']
        meta = loaded.get('meta', {})
    elif isinstance(loaded, dict):
        qtable.table = loaded
        meta = {}
    else:
        qtable.load(qtable_path); meta = {}
    bins_preset = meta.get('bins','original')
    import gymnasium as gym
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    import imageio
    all_frames = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs, bins_preset=bins_preset)
        frames = []
        for t in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            best_action = qtable.get_best_action(state)
            try:
                act_id = ACTION_ID[best_action]
            except Exception:
                act_id = int(best_action)
            obs, reward, done, truncated, info = env.step(act_id)
            state = get_state(obs, bins_preset=bins_preset)
            if done:
                break
        # pad frames to reasonable length or append
        all_frames.extend(frames)
    try: env.close()
    except: pass
    if len(all_frames) == 0:
        print('No frames captured; cannot save GIF')
        return False
    imageio.mimsave(out_path, all_frames, fps=30)
    return True


def main(topcsv, qtables_dir=None, n_eval=100, render_episodes=5):
    if not os.path.exists(topcsv):
        print('topcsv not found:', topcsv); return 2
    df = pd.read_csv(topcsv)
    if qtables_dir is None:
        possible = glob.glob(os.path.join('logs','*','qtables'))
        if len(possible)==0:
            print('No qtables dir'); return 2
        qtables_dir = possible[0]
    print('Using qtables_dir', qtables_dir)
    rows = []
    per_qtable_records = []
    for idx, row in df.iterrows():
        alpha = row['alpha']; gamma=row['gamma']; bins=row['bins']; eps_decay_type=row['eps_decay_type']; eps_start=row['eps_start']; eps_end=row['eps_end']
        matches = find_qtables_for_combo(qtables_dir, alpha, gamma, bins, eps_decay_type, eps_start, eps_end)
        print('Combo', idx, 'found', len(matches), 'qtables')
        for p in matches:
            totals = eval_qtable(p, n_episodes=n_eval)
            per_qtable_records.append({'path': p, 'alpha': alpha, 'gamma': gamma, 'bins': bins, 'eps_decay_type': eps_decay_type, 'eps_start': eps_start, 'eps_end': eps_end, 'n': n_eval, 'mean': float(np.mean(totals)), 'std': float(np.std(totals)), 'totals': totals})
    # save raw
    os.makedirs('figures', exist_ok=True)
    raw_csv = os.path.join('figures', f'top3_detailed_raw_{TS}.csv')
    # totals is list; convert to string
    df_raw = pd.DataFrame([{k:v for k,v in r.items() if k!='totals'} for r in per_qtable_records])
    df_raw.to_csv(raw_csv, index=False)
    print('Saved raw per-qtable summary to', raw_csv)

    # boxplot per combo
    import matplotlib.pyplot as plt
    combos = df[['alpha','gamma','bins','eps_decay_type','eps_start','eps_end']].to_dict('records')
    combo_labels = []
    combo_data = []
    for c in combos:
        cpaths = [r for r in per_qtable_records if r['alpha']==c['alpha'] and r['gamma']==c['gamma'] and r['bins']==c['bins'] and r['eps_decay_type']==c['eps_decay_type'] and r['eps_start']==c['eps_start'] and r['eps_end']==c['eps_end']]
        # flatten totals from each qtable
        totals = []
        for cp in cpaths:
            totals.extend(cp['totals'])
        combo_labels.append(f"a={c['alpha']} g={c['gamma']} b={c['bins']}")
        combo_data.append(totals)
    plt.figure(figsize=(10,6))
    plt.boxplot(combo_data, labels=combo_labels)
    plt.title('Per-episode distribution across seeds (n_eval='+str(n_eval)+')')
    plt.ylabel('Episode reward')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out_box = os.path.join('figures', f'top3_boxplot_{TS}.png')
    plt.savefig(out_box)
    print('Saved boxplot to', out_box)

    # barplot of per-qtable means
    plt.figure(figsize=(12,6))
    paths = [r['path'] for r in per_qtable_records]
    means = [r['mean'] for r in per_qtable_records]
    labels = [os.path.basename(p).replace('.pkl','') for p in paths]
    plt.bar(range(len(means)), means)
    plt.xticks(range(len(means)), labels, rotation=90)
    plt.ylabel('Per-qtable mean (n_eval='+str(n_eval)+')')
    plt.tight_layout()
    out_bar = os.path.join('figures', f'top3_per_qtable_means_{TS}.png')
    plt.savefig(out_bar)
    print('Saved per-qtable barplot to', out_bar)

    # render best qtable (highest mean)
    best = max(per_qtable_records, key=lambda r: r['mean'])
    gif_out = os.path.join('figures', f"best_qtable_render_{TS}.gif")
    ok = render_qtable_gif(best['path'], gif_out, n_episodes=render_episodes)
    if ok:
        print('Saved GIF to', gif_out)
    else:
        print('Failed to save GIF')
    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--topcsv', default='figures/top3_for_eval_1000ep.csv')
    p.add_argument('--qtables_dir', default=None)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--render_episodes', type=int, default=5)
    args = p.parse_args()
    sys.exit(main(args.topcsv, args.qtables_dir, args.n_eval, args.render_episodes))
