import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool

# Basic sweep space (compact to start)
ALPHAS = [0.10, 0.15]
GAMMAS = [0.95, 0.99]
EPS_TYPES = ["exp", "linear"]
EPS_START = [0.5]
EPS_END = [0.01]
EPS_EPISODES = [3000]
RETRY_PENALTY_SCALES = [0.2, 0.3, 0.5]
MAX_RETRIES = [3, 5]
SEEDS = [101, 102]

# Episodes per config for quick profitability screening
DEFAULT_EPISODES = 3000

PYTHON = sys.executable


def run_config(args):
    (alpha, gamma, eps_type, eps_start, eps_end, eps_episodes, rps, mr, seed, episodes, stamp) = args
    cfg_name = f"a{alpha}_g{gamma}_eps{eps_type}_s{eps_start}_e{eps_end}_ep{eps_episodes}_rps{rps}_mr{mr}_seed{seed}"
    out_dir = os.path.join("logs", "sweeps", stamp, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    history_path = os.path.join(out_dir, "history.csv")
    qtable_path = os.path.join(out_dir, "qtable.pkl")
    cmd = [
        PYTHON, "ai_player_Q.py",
        "--alpha", str(alpha),
        "--gamma", str(gamma),
        "--games", str(episodes),
        "--dealer_host", "localhost",
        "--eps_decay_type", eps_type,
        "--eps_start", str(eps_start),
        "--eps_end", str(eps_end),
        "--eps_decay_episodes", str(eps_episodes),
        "--retry_penalty_scale", str(rps),
        "--max_retries_per_game", str(mr),
        "--seed", str(seed),
        "--quiet",
        "--history", history_path,
        "--save", qtable_path,
    ]
    env = os.environ.copy()
    # Ensure dealer is running externally before invoking this script.
    try:
        subprocess.run(cmd, check=True, env=env)
        return {"config": cfg_name, "status": "ok", "out_dir": out_dir}
    except subprocess.CalledProcessError as e:
        return {"config": cfg_name, "status": "fail", "code": e.returncode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_space = list(itertools.product(
        ALPHAS, GAMMAS, EPS_TYPES, EPS_START, EPS_END, EPS_EPISODES,
        RETRY_PENALTY_SCALES, MAX_RETRIES, SEEDS
    ))
    jobs = [(a,g,et,es,ee,epe,rps,mr,seed,args.episodes,stamp) for (a,g,et,es,ee,epe,rps,mr,seed) in sweep_space]

    with Pool(processes=args.workers) as pool:
        results = pool.map(run_config, jobs)

    summary_path = os.path.join("logs", "sweeps", stamp, "sweep_results.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path} ({len(results)} configs)")


if __name__ == "__main__":
    main()
