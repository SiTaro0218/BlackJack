import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool

PYTHON = sys.executable

# Near-best base around earlier results
ALPHAS = [0.10, 0.15]
GAMMAS = [0.99, 0.95]
EPS_DECAY_TYPES = ["exp"]
EPS_START = [0.5]
EPS_END = [0.01]
EPS_DECAY_EPISODES = [3000]
# Retry suppression grid
RETRY_PENALTY_SCALES = [0.0, 0.1, 0.2]
MAX_RETRIES = [0, 1, 2]
SEEDS = [101, 102]

DEFAULT_GAMES = 3000


def run_config(args):
    (alpha, gamma, eps_decay_type, eps_start, eps_end, eps_decay_episodes, rps, mr, seed, games, stamp) = args
    cfg_name = f"retrySupp_a{alpha}_g{gamma}_eps{eps_decay_type}_s{eps_start}_e{eps_end}_ep{eps_decay_episodes}_rps{rps}_mr{mr}_seed{seed}"
    out_dir = os.path.join("logs", "sweeps", stamp, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    history_path = os.path.join(out_dir, "history.csv")
    qtable_path = os.path.join(out_dir, "qtable.pkl")

    cmd = [
        PYTHON, "ai_player_Q.py",
        "--alpha", str(alpha),
        "--gamma", str(gamma),
        "--games", str(games),
        "--dealer_host", "localhost",
        "--eps_decay_type", eps_decay_type,
        "--eps_start", str(eps_start),
        "--eps_end", str(eps_end),
        "--eps_decay_episodes", str(eps_decay_episodes),
        "--retry_penalty_scale", str(rps),
        "--max_retries_per_game", str(mr),
        "--seed", str(seed),
        "--quiet",
        "--history", history_path,
        "--save", qtable_path,
    ]
    env = os.environ.copy()
    try:
        subprocess.run(cmd, check=True, env=env)
        return {"config": cfg_name, "status": "ok", "out_dir": out_dir}
    except subprocess.CalledProcessError as e:
        return {"config": cfg_name, "status": "fail", "code": e.returncode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_retry_supp")
    sweep_space = list(itertools.product(
        ALPHAS, GAMMAS, EPS_DECAY_TYPES, EPS_START, EPS_END, EPS_DECAY_EPISODES,
        RETRY_PENALTY_SCALES, MAX_RETRIES, SEEDS
    ))
    jobs = [(a,g,edt,es,ee,epe,rps,mr,seed,args.games,stamp) for (a,g,edt,es,ee,epe,rps,mr,seed) in sweep_space]

    with Pool(processes=args.workers) as pool:
        results = pool.map(run_config, jobs)

    summary_path = os.path.join("logs", "sweeps", stamp, "sweep_results.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path} ({len(results)} configs)")


if __name__ == "__main__":
    main()
