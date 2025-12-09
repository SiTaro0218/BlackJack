import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUN_RE = re.compile(r"a(?P<alpha>[^_]+)_g(?P<gamma>[^_]+)_s(?P<eps_start>[^_]+)_e(?P<eps_end>[^_]+)_d(?P<decay>[^_]+)_seed(?P<seed>\d+)")


def parse_run_name(run_name: str):
    m = RUN_RE.match(run_name)
    if not m:
        raise ValueError(f"Unparsable run_name: {run_name}")
    gd = m.groupdict()
    return {
        'alpha': gd['alpha'],
        'gamma': gd['gamma'],
        'eps_start': gd['eps_start'],
        'eps_end': gd['eps_end'],
        'decay': gd['decay'],
        'seed': gd['seed'],
    }


def build_cmd(params: dict, games: int, out_run_name: str):
    python_exec = sys.executable or 'python'
    history = ROOT / 'logs' / f'hparam_{out_run_name}.history.csv'
    qtable_dir = ROOT / 'logs' / f'hparam_{out_run_name}' / 'qtables'
    qtable_dir.mkdir(parents=True, exist_ok=True)
    return [
        python_exec, 'ai_player_Q.py',
        '--games', str(games),
        '--eps_decay_episodes', str(games),
        '--alpha', params['alpha'],
        '--gamma', params['gamma'],
        '--eps_start', params['eps_start'],
        '--eps_end', params['eps_end'],
        '--eps_decay_type', params['decay'],
        '--history', str(history),
        '--save', str(qtable_dir / f"{out_run_name}.pkl"),
    ]


def main():
    ap = argparse.ArgumentParser(description='Retrain top-N hyperparameter configurations')
    ap.add_argument('--summary', default=str(ROOT / 'logs' / 'summary.csv'))
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--skip-first', action='store_true', help='Skip current rank 1 (e.g., keep as baseline)')
    ap.add_argument('--runs', nargs='+', help='Explicit run names to retrain (e.g. a0.1_g0.98_s0.5_e0.01_dexp_seed3). Overrides --top selection.')
    ap.add_argument('--games', type=int, default=2000)
    ap.add_argument('--suffix', default='_rerun', help='Suffix added to run_name for retrained runs')
    ap.add_argument('--timeout', type=int, default=0)
    ap.add_argument('--stagger_ms', type=int, default=200)
    args = ap.parse_args()

    rows = []
    with open(args.summary, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get('rank'): continue
            rows.append(r)
    rows = sorted(rows, key=lambda r: int(r['rank']))

    if args.runs:
        # Use explicit list
        selected_stems = args.runs
        print(f"[INFO] Using explicit --runs list ({len(selected_stems)}).")
    else:
        start_idx = 1 if args.skip_first else 0
        target = rows[start_idx:start_idx + args.top]
        if not target:
            print('No rows selected. Exiting.')
            return
        selected_stems = []
        for r in target:
            hist_path = r['history_path']
            stem = Path(hist_path).name.replace('hparam_','').replace('.history.csv','')
            selected_stems.append(stem)
        print(f"[INFO] Selected {len(selected_stems)} runs for retraining (games={args.games}).")

    for stem in selected_stems:
        params = parse_run_name(stem)
        out_run_name = f"{stem}{args.suffix}"
        cmd = build_cmd(params, args.games, out_run_name)
        print('[RETRAIN]', ' '.join(cmd))
        timeout = args.timeout if args.timeout > 0 else None
        rc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout).returncode
        if rc != 0:
            print(f"  -> FAILED rc={rc} for {out_run_name}")
        else:
            print(f"  -> DONE {out_run_name}")

    print('[INFO] Retraining complete.')


if __name__ == '__main__':
    main()
