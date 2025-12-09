import csv
import os
import argparse
from collections import defaultdict
import numpy as np
import concurrent.futures
import sys
import os

# ensure project root (parent of tools/) is on sys.path so we can import run_experiments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import run_experiments


def read_results(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['episodes'] = int(r.get('episodes', '0') or 0)
            try:
                r['mov_avg_100'] = float(r.get('mov_avg_100', '0') or 0)
            except Exception:
                r['mov_avg_100'] = 0.0
            rows.append(r)
    return rows


def group_key_from_row(r):
    return (r['alpha'], r['gamma'], r['eps_start'], r['eps_end'], r['eps_decay_type'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results.csv')
    p.add_argument('--logs', default='logs')
    p.add_argument('--top', type=int, default=3)
    p.add_argument('--min_episodes', type=int, default=900)
    p.add_argument('--games', type=int, default=5000, help='episodes per run')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--save-qtables', action='store_true')
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s!='']

    rows = read_results(args.results)
    long_rows = [r for r in rows if r['episodes'] >= args.min_episodes]
    if not long_rows:
        print('No long runs found in results; will still select top groups by available data')

    groups = defaultdict(list)
    for r in rows:
        key = group_key_from_row(r)
        groups[key].append(r)

    group_scores = []
    for k, rs in groups.items():
        vals = [r['mov_avg_100'] for r in rs if r.get('episodes',0) >= args.min_episodes]
        if not vals:
            # fallback to available seeds
            vals = [r['mov_avg_100'] for r in rs]
        if vals:
            group_scores.append((np.mean(vals), k))

    if not group_scores:
        print('No groups found in results.csv')
        sys.exit(1)

    group_scores.sort(key=lambda t: t[0])
    top = [k for _, k in group_scores[:args.top]]

    print('Top groups:')
    for k in top:
        print(' ', k)

    os.makedirs(args.logs, exist_ok=True)
    qtables_dir = os.path.join(args.logs, 'qtables') if args.save_qtables else None
    if qtables_dir:
        os.makedirs(qtables_dir, exist_ok=True)

    # prepare params list
    all_params = []
    for key in top:
        alpha, gamma, eps_start, eps_end, decay = key
        for s in seeds:
            params = {'alpha': float(alpha), 'gamma': float(gamma), 'eps_start': float(eps_start), 'eps_end': float(eps_end), 'eps_decay_type': decay, 'seed': int(s), 'eps_decay_episodes': 1000}
            all_params.append(params)

    print(f'Launching {len(all_params)} runs with {args.workers} workers, {args.games} episodes each')

    fieldnames = ['run_name','alpha','gamma','eps_start','eps_end','eps_decay_type','seed','episodes','avg_reward','mov_avg_100','logfile']
    write_header = not os.path.exists(args.results)

    results_out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_experiments.run_one, sys.executable, os.path.join(os.path.dirname(run_experiments.__file__), 'ai_player_Q.py'), params, args.logs, args.games, None, qtables_dir): params for params in all_params}
        with open(args.results, 'a', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for fut in concurrent.futures.as_completed(futures):
                params = futures[fut]
                try:
                    res = fut.result()
                    results_out.append(res)
                    writer.writerow({k: res.get(k) for k in fieldnames})
                    csvf.flush()
                    print('Recorded', res['run_name'])
                except Exception as e:
                    print('Run failed for', params, e)

    print('Launched and recorded long runs.')


if __name__ == '__main__':
    main()
