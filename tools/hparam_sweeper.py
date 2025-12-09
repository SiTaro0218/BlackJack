import argparse
import itertools
import os
import subprocess
import time
import sys


def run(cmd: list[str], timeout: int | None = None):
    try:
        # Stream output to terminal; avoid capture to prevent large buffer stalls
        return subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=124)


def main():
    parser = argparse.ArgumentParser(description='Wide hyperparameter sweep orchestrator')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.9, 0.95, 0.98])
    parser.add_argument('--eps_start', type=float, nargs='+', default=[1.0, 0.5])
    parser.add_argument('--eps_end', type=float, nargs='+', default=[0.01, 0.05])
    parser.add_argument('--eps_decay_type', type=str, nargs='+', default=['const', 'linear', 'exp'])
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--stagger_ms', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=0, help='per-run timeout in seconds; 0 disables')
    parser.add_argument('--save_qtables', action='store_true')
    args = parser.parse_args()

    # Ensure dealer is running separately; here we just remind
    print('[INFO] Please ensure dealer.py is running persistently in another terminal.')

    # Directly run ai_player_Q.py per combination and seed (since run_experiments.py doesn't accept these args)
    combos = list(itertools.product(args.alphas, args.gammas, args.eps_start, args.eps_end, args.eps_decay_type))
    print(f'[INFO] Total combinations: {len(combos)} x seeds {len(args.seeds)}')

    for (alpha, gamma, eps_s, eps_e, decay) in combos:
        for seed in args.seeds:
            time.sleep(args.stagger_ms / 1000.0)
            run_name = f"a{alpha}_g{gamma}_s{eps_s}_e{eps_e}_d{decay}_seed{seed}"
            history = os.path.join('logs', f'hparam_{run_name}.history.csv')
            qtable_dir = os.path.join('logs', f'hparam_{run_name}', 'qtables')
            os.makedirs(qtable_dir, exist_ok=True)
            # Use the current Python executable (venv-aware)
            python_exec = sys.executable or "python"
            cmd = [
                python_exec, 'ai_player_Q.py',
                '--games', str(args.games),
                '--eps_decay_episodes', str(args.games),
                '--alpha', str(alpha),
                '--gamma', str(gamma),
                '--eps_start', str(eps_s),
                '--eps_end', str(eps_e),
                '--eps_decay_type', str(decay),
                '--history', history,
                '--save', os.path.join(qtable_dir, f'{run_name}.pkl'),
            ]
            print('[RUN]', ' '.join(cmd))
            timeout = args.timeout if args.timeout and args.timeout > 0 else None
            res = run(cmd, timeout=timeout)
            if res.returncode != 0:
                if res.returncode == 124:
                    print(f'[TIMEOUT] {run_name} exceeded {timeout}s, skipping.')
                else:
                    print(f'[ERR] returncode={res.returncode} for {run_name}')


if __name__ == '__main__':
    main()
