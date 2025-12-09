"""Evaluate qtable pickles for top-K parameter combinations.
Usage: python scripts/eval_topk_qtables.py --topcsv figures/top10_by_mean_*.csv --results results_autocollect_FULL_*.csv
"""
import os
import sys
import glob
import argparse
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt

TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Import CartPole helpers (get_state, select_action, q_table, Strategy, GAME_NAME)
import sys
import os
# ensure repo root is on sys.path so we can import local modules (classes.py)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from enum import Enum
import numpy as np
from classes import Strategy, QTable

# We'll implement a small subset of CartPole_v1 functionality here so we don't need
# to import CartPole_v1 (which pulls gymnasium at module import time).
GAME_NAME = 'CartPole-v1'


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


def run_eval(qtable_path, n_episodes=10, max_steps=500):
    # load qtable
    try:
        with open(qtable_path, 'rb') as f:
            loaded = pickle.load(f)
    except Exception as e:
        print('Failed to load', qtable_path, e)
        return None
    # prepare a fresh QTable instance for evaluation
    qtable = QTable(action_class=Action, default_value=0)
    if isinstance(loaded, dict) and 'table' in loaded:
        qtable.table = loaded['table']
        meta = loaded.get('meta', {})
    elif isinstance(loaded, dict):
        qtable.table = loaded
        meta = {}
    else:
        try:
            qtable.load(qtable_path)
            meta = {}
        except Exception as e:
            print('Unknown qtable format for', qtable_path, e)
            return None

    bins_preset = meta.get('bins', 'original')

    # create environment (import gymnasium lazily)
    try:
        import gymnasium as gym
    except Exception as e:
        print('gymnasium is required for evaluation but not available:', e)
        return None
    env = gym.make(GAME_NAME)

    totals = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs, bins_preset=bins_preset)
        total = 0
        done = False
        for t in range(max_steps):
            # greedy: pick best action from qtable
            # QTable.get_best_action expects an Action Enum class passed at construction
            best_action = qtable.get_best_action(state)
            # best_action is an Action enum
            try:
                act_id = ACTION_ID[best_action]
            except Exception:
                # fallback if qtable stored ints
                act_id = int(best_action)
            obs, reward, done, truncated, info = env.step(act_id)
            state = get_state(obs, bins_preset=bins_preset)
            total += reward
            if done:
                break
        totals.append(total)
    try:
        env.close()
    except Exception:
        pass
    return {'n': n_episodes, 'mean': float(pd.Series(totals).mean()), 'std': float(pd.Series(totals).std()), 'runs': totals}


def float_to_token(x):
    s = str(x)
    return s.replace('.', '_')


def find_qtables_for_combo(qtables_dir, alpha, gamma, bins, eps_decay_type, eps_start, eps_end):
    a = float_to_token(alpha)
    g = float_to_token(gamma)
    es = float_to_token(eps_start)
    ee = float_to_token(eps_end)
    pattern = os.path.join(qtables_dir, f"*alpha-{a}--gamma-{g}--eps_start-{es}--eps_end-{ee}--eps_decay_type-{eps_decay_type}--bins-{bins}--seed-*.pkl")
    matches = glob.glob(pattern)
    return sorted(matches)


def main(topcsv, results_csv=None, qtables_dir=None, n_episodes=10):
    if not os.path.exists(topcsv):
        print('topcsv not found:', topcsv)
        return 2
    topdf = pd.read_csv(topcsv)

    # locate qtables_dir if not provided: search for logs/*/qtables
    if qtables_dir is None:
        possible = glob.glob(os.path.join('logs', '*', 'qtables'))
        if len(possible) == 0:
            print('No qtables dir found under logs/*/qtables')
            return 2
        qtables_dir = possible[0]
    print('Using qtables_dir:', qtables_dir)

    rows = []
    for idx, row in topdf.iterrows():
        alpha = row['alpha']
        gamma = row['gamma']
        bins = row['bins']
        eps_decay_type = row['eps_decay_type']
        eps_start = row.get('eps_start', row.get('eps_start'))
        eps_end = row.get('eps_end', row.get('eps_end'))

        matches = find_qtables_for_combo(qtables_dir, alpha, gamma, bins, eps_decay_type, eps_start, eps_end)
        if len(matches) == 0:
            print('No qtables found for combo:', alpha, gamma, bins, eps_decay_type, eps_start, eps_end)
            rows.append({'alpha':alpha,'gamma':gamma,'bins':bins,'eps_decay_type':eps_decay_type,'eps_start':eps_start,'eps_end':eps_end,'found':0,'mean':None,'std':None})
            continue
        # evaluate each matched qtable
        combo_means = []
        combo_std = []
        for p in matches:
            res = run_eval(p, n_episodes=n_episodes)
            if res is None:
                continue
            combo_means.append(res['mean'])
            combo_std.append(res['std'])
        if len(combo_means) == 0:
            rows.append({'alpha':alpha,'gamma':gamma,'bins':bins,'eps_decay_type':eps_decay_type,'eps_start':eps_start,'eps_end':eps_end,'found':len(matches),'mean':None,'std':None})
            continue
        rows.append({'alpha':alpha,'gamma':gamma,'bins':bins,'eps_decay_type':eps_decay_type,'eps_start':eps_start,'eps_end':eps_end,'found':len(matches),'mean':float(pd.Series(combo_means).mean()),'std':float(pd.Series(combo_means).std())})

    out_df = pd.DataFrame(rows)
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir('figures')
    out_csv = os.path.join('figures', f'top10_evals_{TS}.csv')
    out_df.to_csv(out_csv, index=False)
    print('Saved evaluations to', out_csv)

    # bar plot
    plot_df = out_df.dropna(subset=['mean']).sort_values('mean', ascending=False)
    if not plot_df.empty:
        labels = plot_df.apply(lambda r: f"a={r['alpha']} g={r['gamma']} b={r['bins']} d={r['eps_decay_type']}", axis=1)
        plt.figure(figsize=(10,5))
        plt.bar(range(len(plot_df)), plot_df['mean'], yerr=plot_df['std'].fillna(0), capsize=4)
        plt.xticks(range(len(plot_df)), labels, rotation=45, ha='right')
        plt.ylabel('mean eval reward')
        plt.title('Top combos eval (per-combo mean of seed-wise eval means)')
        plt.tight_layout()
        out_png = os.path.join('figures', f'top10_evals_{TS}.png')
        plt.savefig(out_png)
        print('Saved plot to', out_png)

    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--topcsv', default=None)
    p.add_argument('--results', default=None)
    p.add_argument('--qtables_dir', default=None)
    p.add_argument('--n_episodes', type=int, default=10)
    args = p.parse_args()
    if args.topcsv is None:
        # try to find top10 file
        files = glob.glob(os.path.join('figures', 'top10_by_mean_*.csv'))
        if not files:
            print('No top10 csv found in figures/')
            sys.exit(2)
        args.topcsv = files[0]
    sys.exit(main(args.topcsv, args.results, args.qtables_dir, args.n_episodes))
