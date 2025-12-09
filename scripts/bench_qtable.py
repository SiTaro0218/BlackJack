"""Quick benchmark: run a single qtable for N episodes and print timing.
Usage: python scripts/bench_qtable.py PATH_TO_QTABLE --n_episodes 10
"""
import argparse
import time
import pickle
import os
import sys

# ensure repo root on path for classes
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from classes import QTable
from enum import Enum
import numpy as np

TS = None

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


def run_bench(qtable_path, n_episodes=10, max_steps=500):
    try:
        with open(qtable_path,'rb') as f:
            loaded = pickle.load(f)
    except Exception as e:
        print('Failed to load', qtable_path, e)
        return None
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
            print('Unknown qtable format', e)
            return None

    bins_preset = meta.get('bins', 'original')

    try:
        import gymnasium as gym
    except Exception as e:
        print('gymnasium not available:', e)
        return None

    env = gym.make('CartPole-v1')
    totals = []
    start = time.time()
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs, bins_preset=bins_preset)
        total = 0
        done = False
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
    end = time.time()
    try:
        env.close()
    except Exception:
        pass
    elapsed = end - start
    return {'n': n_episodes, 'elapsed_sec': elapsed, 'per_episode': elapsed / n_episodes, 'totals': totals}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('qtable')
    p.add_argument('--n_episodes', type=int, default=10)
    args = p.parse_args()
    res = run_bench(args.qtable, n_episodes=args.n_episodes)
    if res is None:
        sys.exit(2)
    print('Bench result:', res)
    # also print human-friendly
    print(f"Total time: {res['elapsed_sec']:.2f}s, per-episode: {res['per_episode']:.3f}s")
