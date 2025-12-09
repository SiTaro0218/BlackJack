import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def qtable_from_history_path(history_path: str) -> Path:
    hp = Path(history_path)
    # history filename example: hparam_a0.1_g0.98_s0.5_e0.01_dexp_seed3.history.csv
    name = hp.name
    if not name.startswith('hparam_'):
        raise ValueError(f'Unexpected history file name: {name}')
    stem = name[:-4] if name.endswith('.csv') else name
    stem = stem.replace('.history', '')  # -> hparam_a0.1_...
    run_dir = ROOT / 'logs' / stem
    qdir = run_dir / 'qtables'
    # qtable file example: a0.1_g0.98_s0.5_e0.01_dexp_seed3.pkl
    candidate = stem.replace('hparam_', '') + '.pkl'
    qpath = qdir / candidate
    if qpath.exists():
        return qpath
    # fallback: pick first .pkl under qtables
    pkls = list(qdir.glob('*.pkl'))
    if pkls:
        return pkls[0]
    raise FileNotFoundError(f'QTable not found under {qdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default=str(ROOT / 'logs' / 'summary.csv'))
    ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--skip-first', action='store_true', help='Skip rank 1 entry')
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()

    rows = []
    with open(args.summary, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get('rank'):
                continue
            rows.append(r)

    # sort by integer rank just in case
    rows = sorted(rows, key=lambda r: int(r['rank']))

    start_idx = 1 if args.skip_first else 0
    target = rows[start_idx:start_idx + args.top]

    if not target:
        print('No entries selected from summary.')
        sys.exit(1)

    py = sys.executable
    for r in target:
        hist = r['history_path']
        qtable = qtable_from_history_path(hist)
        tag = qtable.stem  # a0.1_g0.98_...
        outdir = ROOT / 'logs' / f'eval_multi_{tag}'
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"== Evaluating rank {r['rank']}: win_rate={float(r['win_rate']):.3f}, avg_reward={float(r['avg_reward']):.3f}")
        print(f'QTable: {qtable}')
        cmd = [
            py,
            str(ROOT / 'tools' / 'multi_evaluate.py'),
            '--load', str(qtable),
            '--games', str(args.games),
            '--repeats', str(args.repeats),
            '--outdir', str(outdir),
        ]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f'  -> multi_evaluate failed with exit code {rc}')
        else:
            print(f'  -> Done: {outdir}')


if __name__ == '__main__':
    main()
