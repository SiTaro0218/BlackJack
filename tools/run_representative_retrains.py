import os
import subprocess
import time
import argparse

"""Sequential long retrains with extended state & RETRY penalties.
Adds logging capture and --games override for quick smoke tests.
"""

# Representative hyperparameter sets (tunable)
REP_SETS = [
    {"alpha": 0.2, "gamma": 0.98, "eps_start": 0.5, "eps_end": 0.01, "eps_decay_episodes": 4000, "eps_decay_type": "exp", "retry_penalty_scale": 0.5, "max_retries_per_game": 5, "seed": 101},
    {"alpha": 0.3, "gamma": 0.9,  "eps_start": 0.5, "eps_end": 0.01, "eps_decay_episodes": 4000, "eps_decay_type": "exp", "retry_penalty_scale": 0.3, "max_retries_per_game": 5, "seed": 102},
    {"alpha": 0.1, "gamma": 0.98, "eps_start": 0.5, "eps_end": 0.01, "eps_decay_episodes": 4000, "eps_decay_type": "exp", "retry_penalty_scale": 0.7, "max_retries_per_game": 5, "seed": 103},
]


def build_name(p):
    return (
        f"rep_a{p['alpha']}_g{p['gamma']}_s{p['eps_start']}_e{p['eps_end']}"
        f"_d{p['eps_decay_type']}_rps{p['retry_penalty_scale']}_mr{p['max_retries_per_game']}_seed{p['seed']}"
    )


def run_set(p, venv_python, games):
    name = build_name(p)
    run_dir = os.path.join('logs', 'extended_runs', name)
    qdir = os.path.join(run_dir, 'qtables')
    os.makedirs(qdir, exist_ok=True)
    history_path = os.path.join(run_dir, f"{name}.history.csv")
    qtable_path = os.path.join(qdir, f"{name}.pkl")
    log_path = os.path.join(run_dir, f"{name}.stdout.txt")
    cmd = [
        venv_python, 'ai_player_Q.py',
        '--games', str(games),
        '--alpha', str(p['alpha']),
        '--gamma', str(p['gamma']),
        '--eps_start', str(p['eps_start']),
        '--eps_end', str(p['eps_end']),
        '--eps_decay_episodes', str(p['eps_decay_episodes']),
        '--eps_decay_type', p['eps_decay_type'],
        '--retry_penalty_scale', str(p['retry_penalty_scale']),
        '--max_retries_per_game', str(p['max_retries_per_game']),
        '--history', history_path,
        '--save', qtable_path,
        '--seed', str(p['seed']),
        '--quiet'
    ]
    print(f"[RUN] {name} -> {history_path} (games={games})")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        with open(log_path, 'w', encoding='utf-8') as lf:
            lf.write(proc.stdout)
            lf.write("\n---- STDERR ----\n")
            lf.write(proc.stderr)
        if proc.returncode != 0:
            print(f"[FAIL] {name} rc={proc.returncode} see {log_path}")
        else:
            print(f"[DONE] {name} rc=0")
    except Exception as e:
        print(f"[EXCEPTION] {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run representative extended-state retrains sequentially.')
    parser.add_argument('--skip', type=str, default='', help='Comma-separated substrings; if name contains any, skip that set.')
    parser.add_argument('--python', type=str, default=os.path.join('.venv','Scripts','python.exe'))
    parser.add_argument('--games', type=int, default=10000, help='Number of games per run (e.g. 200 for smoke test)')
    args = parser.parse_args()
    skips = [s.strip() for s in args.skip.split(',') if s.strip()]

    abs_python = os.path.abspath(args.python)
    for p in REP_SETS:
        name = build_name(p)
        if any(s in name for s in skips):
            print(f"[SKIP] {name}")
            continue
        run_set(p, abs_python, args.games)

    print('All representative retrains finished.')


if __name__ == '__main__':
    main()
