#!/usr/bin/env python3
"""Experiment runner for ai_player_Q.py (BlackJack)

Creates logs/<run_name>.txt and a summary CSV results.csv with basic metrics.

Usage:
    python run_experiments.py --quick --script ai_player_Q.py
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
import csv
import re
import concurrent.futures


def make_run_name(params):
    parts = []
    for k in ['alpha', 'gamma', 'eps_start', 'eps_end', 'eps_decay_type', 'seed']:
        parts.append(f"{k}={params.get(k)}")
    name = "--".join(parts)
    # safe file name
    return name.replace(' ', '').replace('.', '_').replace('/', '_').replace('=', '-')


def parse_rewards_from_log_ai(text):
    # Parse logs produced by ai_player_Q.py. It prints "Game N start." and later prints money lines.
    # We compute per-game reward as (money_at_end - money_at_start) for each game block.
    rewards = []
    lines = text.splitlines()
    current_game = None
    start_money = None
    last_money = None
    for ln in lines:
        ln = ln.strip()
        m_start = re.match(r'^Game\s+(\d+)\s+start\.', ln)
        if m_start:
            # start of a new game
            current_game = int(m_start.group(1))
            start_money = None
            last_money = None
            continue
        # money lines: e.g. "money:  10000 $" or "money: 10020 $"
        m_money = re.search(r'money:\s*([0-9]+)', ln)
        if m_money and current_game is not None:
            val = float(m_money.group(1))
            if start_money is None:
                start_money = val
            last_money = val
            continue
        # end of game marker
        if ln.startswith('Game finished.') and current_game is not None:
            if start_money is not None and last_money is not None:
                rewards.append(last_money - start_money)
            # reset
            current_game = None
            start_money = None
            last_money = None

    return rewards


def moving_average(arr, window=100):
    if not arr:
        return 0.0
    n = len(arr)
    if n < window:
        return sum(arr) / n
    return sum(arr[-window:]) / min(window, n)


def run_one(py_exec, script_path, params, out_dir, games, timeout=None, qtables_dir=None):
    run_name = make_run_name(params)
    out_path = os.path.join(out_dir, run_name + '.txt')

    # Only support ai_player_Q (BlackJack) for this repo's runner.
    history_path = os.path.join(out_dir, run_name + '.history.csv')
    cmd = [py_exec, script_path,
           '--games', str(games),
           '--history', history_path,
           '--alpha', str(params['alpha']),
           '--gamma', str(params['gamma']),
           '--eps_start', str(params['eps_start']),
           '--eps_end', str(params['eps_end']),
           '--eps_decay_episodes', str(params.get('eps_decay_episodes', 1000)),
           '--eps_decay_type', params['eps_decay_type']]
    if qtables_dir is not None:
        save_path = os.path.join(qtables_dir, run_name + '.pkl')
        cmd += ['--save', save_path]

    # run
    print(f"Running: {run_name} -> {out_path}")
    start = time.time()
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        elapsed = time.time() - start
        print(f"Finished in {elapsed:.1f}s, returncode={proc.returncode}")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {run_name}")

    # read back and parse
    text = ''
    # read log robustly: try utf-8, then cp932 (Windows), then fallback to binary decode with replacement
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(out_path, 'r', encoding='cp932', errors='replace') as f:
                text = f.read()
        except Exception as e:
            try:
                with open(out_path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='replace')
            except Exception as e2:
                print('Failed to read log (binary fallback):', e2)
    except Exception as e:
        print('Failed to read log:', e)

    # parse using ai_player_Q parser
    rewards = parse_rewards_from_log_ai(text)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    last_mov = moving_average(rewards, window=100)
    return {
        'run_name': run_name,
        'alpha': params['alpha'],
        'gamma': params['gamma'],
        'eps_start': params['eps_start'],
        'eps_end': params['eps_end'],
        'eps_decay_type': params['eps_decay_type'],
        'seed': params.get('seed', ''),
        'episodes': len(rewards),
        'avg_reward': avg_reward,
        'mov_avg_100': last_mov,
        'logfile': out_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a quick small grid for smoke test')
    parser.add_argument('--out_dir', default='logs', help='Directory to save logs')
    parser.add_argument('--results', default='results.csv', help='CSV summary path')
    parser.add_argument('--games', type=int, default=200, help='games per run (episodes)')
    parser.add_argument('--max_steps', type=int, default=200, help='max steps per episode')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1)), help='number of parallel workers')
    parser.add_argument('--seeds', type=str, default='0', help='comma-separated seeds, e.g. "0,1,2"')
    parser.add_argument('--save-qtables', action='store_true', help='save Q-table for each run into <out_dir>/qtables')
    parser.add_argument('--alphas', type=str, default=None, help='Optional comma-separated alpha values to override default grid, e.g. "0.1,0.2,0.3"')
    parser.add_argument('--stagger-ms', type=int, default=0, help='Milliseconds to wait between submitting each job to avoid startup bursts')
    parser.add_argument('--timeout', type=int, default=0, help='Timeout in seconds for each run subprocess; 0 means no timeout')

    py_exec = sys.executable
    parser.add_argument('--script', default='CartPole_v1.py', help='script to run for experiments (file name in same dir)')
    args = parser.parse_args()

    # optional override of alpha grid via --alphas
    parsed_alphas = None
    if args.alphas:
        try:
            parsed_alphas = [float(x) for x in args.alphas.split(',') if x != '']
        except Exception:
            print('Failed to parse --alphas; expected comma-separated floats like "0.1,0.2"')
            sys.exit(1)

    script_path = os.path.join(os.path.dirname(__file__), args.script)

    # fallback: if default CartPole not found but ai_player_Q exists, use it (convenience for this repo)
    if not os.path.exists(script_path):
        alt = os.path.join(os.path.dirname(__file__), 'ai_player_Q.py')
        if args.script == 'CartPole_v1.py' and os.path.exists(alt):
            print('CartPole_v1.py not found; falling back to ai_player_Q.py')
            script_path = alt
        else:
            print(f'{args.script} not found in same directory as runner')
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # parse seeds
    seeds = [int(s) for s in args.seeds.split(',') if s != '']

    if args.quick:
        # quick mode baseline, but allow override
        alphas = parsed_alphas if parsed_alphas is not None else [0.1]
        gammas = [0.98]
        eps_starts = [1.0]
        eps_ends = [0.05]
        decay_types = ['linear']
        seeds = seeds or [0]
        games = 20
    else:
        # Expanded grid for more thorough exploration (but allow --alphas override)
        alphas = parsed_alphas if parsed_alphas is not None else [0.01, 0.03, 0.05, 0.1]
        gammas = [0.9, 0.95, 0.98]
        eps_starts = [1.0, 0.5]
        eps_ends = [0.01, 0.05]
        decay_types = ['const', 'linear', 'exp']
        seeds = seeds or [0]
        games = args.games

    max_steps = args.max_steps
    all_params = []
    for a, g, es, ee, dt, s in itertools.product(alphas, gammas, eps_starts, eps_ends, decay_types, seeds):
        all_params.append({'alpha': a, 'gamma': g, 'eps_start': es, 'eps_end': ee, 'eps_decay_type': dt, 'seed': s, 'eps_decay_episodes': 1000})

    results = []
    write_header = not os.path.exists(args.results)
    fieldnames = ['run_name','alpha','gamma','eps_start','eps_end','eps_decay_type','seed','episodes','avg_reward','mov_avg_100','logfile']

    # Run experiments in parallel using ThreadPoolExecutor (subprocesses are spawned per job)
    # prepare qtables dir if requested
    qtables_dir = None
    if args.save_qtables:
        qtables_dir = os.path.join(args.out_dir, 'qtables')
        os.makedirs(qtables_dir, exist_ok=True)

    # determine timeout to pass to run_one (None means no timeout)
    timeout_arg = None if args.timeout == 0 else int(args.timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # run_one signature: (py_exec, script_path, params, out_dir, games, timeout=None, qtables_dir=None)
        future_to_params = {}
        stagger = float(args.stagger_ms) / 1000.0
        for params in all_params:
            fut = executor.submit(run_one, py_exec, script_path, params, args.out_dir, games, timeout_arg, qtables_dir)
            future_to_params[fut] = params
            if stagger > 0:
                time.sleep(stagger)

        with open(args.results, 'a', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            try:
                for fut in concurrent.futures.as_completed(future_to_params):
                    params = future_to_params[fut]
                    try:
                        res = fut.result()
                        results.append(res)
                        writer.writerow({k: res.get(k) for k in fieldnames})
                        csvf.flush()
                        print(f"Recorded result for {res['run_name']}")
                    except Exception as exc:
                        print(f"Run generated an exception: {exc}")
            except KeyboardInterrupt:
                print('\nInterrupted by user (KeyboardInterrupt). Attempting graceful shutdown...')
                # try to cancel not-yet-started futures
                cancelled = 0
                for fut, params in list(future_to_params.items()):
                    if fut.cancel():
                        cancelled += 1
                print(f"Cancelled {cancelled} pending jobs. Waiting for running jobs to finish or exit.")
                # collect results from already completed futures
                for fut in concurrent.futures.as_completed(future_to_params):
                    if fut.done():
                        try:
                            res = fut.result()
                            results.append(res)
                            writer.writerow({k: res.get(k) for k in fieldnames})
                            csvf.flush()
                            print(f"Recorded result for {res['run_name']}")
                        except Exception as exc:
                            print(f"Run generated an exception after interrupt: {exc}")

    print('All runs finished. Summary written to', args.results)


if __name__ == '__main__':
    main()
