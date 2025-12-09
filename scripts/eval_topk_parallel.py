"""Parallel evaluator for qtable pickles.
Reads a topcsv (like figures/top3_for_eval_1000ep.csv), finds qtable files for each combo,
then evaluates each qtable independently (greedy) for n_episodes and aggregates per-combo means.

Usage: python scripts/eval_topk_parallel.py --topcsv figures/top3_for_eval_1000ep.csv --n_episodes 1000 --workers 2
"""
import os
import sys
import glob
import argparse
import datetime
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# prepare repo path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from enum import Enum
import numpy as np
from classes import QTable

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


def eval_one(qtable_path, n_episodes=1000, max_steps=500):
    # run greedy evaluation for one qtable file; return mean/std and n
    try:
        with open(qtable_path,'rb') as f:
            loaded = pickle.load(f)
    except Exception as e:
        return {'path': qtable_path, 'n':0, 'mean': None, 'std': None, 'error': str(e)}

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
            return {'path': qtable_path, 'n':0, 'mean': None, 'std': None, 'error': str(e)}

    bins_preset = meta.get('bins','original')

    try:
        import gymnasium as gym
    except Exception as e:
        return {'path': qtable_path, 'n':0, 'mean': None, 'std': None, 'error': 'gymnasium missing: '+str(e)}

    env = gym.make(GAME_NAME)
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
    try:
        env.close()
    except Exception:
        pass
    return {'path': qtable_path, 'n': n_episodes, 'mean': float(pd.Series(totals).mean()), 'std': float(pd.Series(totals).std())}


def main(topcsv, qtables_dir=None, n_episodes=1000, workers=2):
    if not os.path.exists(topcsv):
        print('topcsv not found:', topcsv)
        return 2
    topdf = pd.read_csv(topcsv)

    if qtables_dir is None:
        possible = glob.glob(os.path.join('logs','*','qtables'))
        if len(possible) == 0:
            print('No qtables dir found under logs/*/qtables')
            return 2
        qtables_dir = possible[0]
    print('Using qtables_dir:', qtables_dir)

    # gather all qtable paths for the combos in topcsv (preserve grouping)
    combo_to_paths = []
    for idx, row in topdf.iterrows():
        alpha = row['alpha']
        gamma = row['gamma']
        bins = row['bins']
        eps_decay_type = row['eps_decay_type']
        eps_start = row.get('eps_start', row.get('eps_start'))
        eps_end = row.get('eps_end', row.get('eps_end'))
        matches = find_qtables_for_combo(qtables_dir, alpha, gamma, bins, eps_decay_type, eps_start, eps_end)
        combo_to_paths.append({'row': row.to_dict(), 'paths': matches})

    # flatten tasks
    tasks = []
    for combo in combo_to_paths:
        for p in combo['paths']:
            tasks.append({'combo': combo['row'], 'path': p})

    if len(tasks) == 0:
        print('No qtables found to evaluate')
        return 2

    print(f'Will evaluate {len(tasks)} qtables using {workers} workers, {n_episodes} eps each')

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(eval_one, t['path'], n_episodes): t for t in tasks}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {'path': task['path'], 'n':0, 'mean': None, 'std': None, 'error': str(e)}
            # attach combo info
            for k,v in task['combo'].items():
                r[k] = v
            results.append(r)
            print('Completed', os.path.basename(r['path']), 'mean=', r.get('mean'), 'err=', r.get('error'))

    # aggregate per combo
    rows = []
    for combo in combo_to_paths:
        combo_paths = combo['paths']
        combo_means = [r['mean'] for r in results if r['path'] in combo_paths and r.get('mean') is not None]
        combo_std = [r['std'] for r in results if r['path'] in combo_paths and r.get('std') is not None]
        rows.append({**combo['row'], 'found': len(combo_paths), 'mean': float(pd.Series(combo_means).mean()) if len(combo_means)>0 else None, 'std': float(pd.Series(combo_means).std()) if len(combo_means)>0 else None})

    out_df = pd.DataFrame(rows)
    os.makedirs('figures', exist_ok=True)
    out_csv = os.path.join('figures', f'top3_evals_parallel_{TS}.csv')
    out_df.to_csv(out_csv, index=False)
    print('Saved aggregated eval to', out_csv)

    # plot
    plot_df = out_df.dropna(subset=['mean']).sort_values('mean', ascending=False)
    if not plot_df.empty:
        import matplotlib.pyplot as plt
        labels = plot_df.apply(lambda r: f"a={r['alpha']} g={r['gamma']} b={r['bins']} d={r['eps_decay_type']}", axis=1)
        plt.figure(figsize=(10,5))
        plt.bar(range(len(plot_df)), plot_df['mean'], yerr=plot_df['std'].fillna(0), capsize=4)
        plt.xticks(range(len(plot_df)), labels, rotation=45, ha='right')
        plt.ylabel('mean eval reward')
        plt.title('Top combos eval (parallel)')
        plt.tight_layout()
        out_png = os.path.join('figures', f'top3_evals_parallel_{TS}.png')
        plt.savefig(out_png)
        print('Saved plot to', out_png)

    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--topcsv', default=None)
    p.add_argument('--qtables_dir', default=None)
    p.add_argument('--n_episodes', type=int, default=1000)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.topcsv is None:
        files = glob.glob(os.path.join('figures', 'top10_by_mean_*.csv'))
        if not files:
            print('No top10 csv found in figures/')
            sys.exit(2)
        args.topcsv = files[0]
    sys.exit(main(args.topcsv, args.qtables_dir, args.n_episodes, args.workers))
