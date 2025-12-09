import argparse
import glob
import os
import sys
import pickle

# Reuse functions from existing export module
from export_policy_table import load_qtable, export_policy

def find_qtables(pattern: str):
    return glob.glob(pattern, recursive=True)


def main():
    parser = argparse.ArgumentParser(description='Batch export policy CSVs for multiple Q-table pickles (rerun models)')
    parser.add_argument('--pattern', type=str, default=os.path.join('logs', '**', 'qtables', '*_rerun.pkl'), help='Glob pattern to locate QTable pickle files')
    parser.add_argument('--outdir', type=str, default='policies_rerun', help='Directory to write CSV outputs')
    parser.add_argument('--softmax_temp', type=float, default=0.5, help='Temperature for softmax policy export')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon for epsilon-greedy policy export')
    parser.add_argument('--skip_softmax', action='store_true', help='Skip softmax export')
    parser.add_argument('--skip_epsilon', action='store_true', help='Skip epsilon-greedy export')
    args = parser.parse_args()

    qtable_paths = find_qtables(args.pattern)
    if not qtable_paths:
        print('No QTable files matched pattern:', args.pattern)
        return

    os.makedirs(args.outdir, exist_ok=True)

    for path in sorted(qtable_paths):
        try:
            qdata = load_qtable(path)
            # Accept meta format {'meta':..., 'table':...}
            if isinstance(qdata, dict) and 'table' in qdata:
                table = qdata['table']
            else:
                table = qdata
            base = os.path.splitext(os.path.basename(path))[0]
            if not args.skip_softmax:
                out_soft = os.path.join(args.outdir, f'{base}_softmax_T{args.softmax_temp}.csv')
                export_policy(table, method='softmax', param=args.softmax_temp, out_csv=out_soft)
                print(f'[OK] softmax exported: {out_soft}')
            if not args.skip_epsilon:
                out_eps = os.path.join(args.outdir, f'{base}_eps{args.epsilon}.csv')
                export_policy(table, method='epsilon_greedy', param=args.epsilon, out_csv=out_eps)
                print(f'[OK] epsilon-greedy exported: {out_eps}')
        except Exception as e:
            print(f'[WARN] failed exporting for {path}: {e}')

    print('Batch export complete. Files in', args.outdir)


if __name__ == '__main__':
    main()
