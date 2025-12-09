import argparse
import glob
import os
import pickle
import sys
import os
from typing import Dict, Tuple, Any

import math


def softmax(q_values, temperature: float):
    if temperature <= 0:
        # fall back to argmax deterministic
        probs = [0.0] * len(q_values)
        max_idx = max(range(len(q_values)), key=lambda i: q_values[i])
        probs[max_idx] = 1.0
        return probs
    # numerical stability
    max_q = max(q_values)
    exps = [math.exp((q - max_q) / temperature) for q in q_values]
    s = sum(exps)
    return [e / s for e in exps]


def epsilon_greedy(q_values, epsilon: float):
    n = len(q_values)
    if epsilon <= 0:
        probs = [0.0] * n
        probs[max(range(n), key=lambda i: q_values[i])] = 1.0
        return probs
    if epsilon >= 1:
        return [1.0 / n] * n
    best = max(range(n), key=lambda i: q_values[i])
    probs = [epsilon / n] * n
    probs[best] += 1 - epsilon
    return probs


def load_qtable(path: str) -> Dict[Any, Any]:
    # Ensure we can import project modules when unpickling (classes)
    proj_root = os.path.dirname(os.path.dirname(__file__))
    if proj_root not in sys.path:
        sys.path.append(proj_root)
    with open(path, 'rb') as f:
        return pickle.load(f)


def export_policy(qtable: Dict[Any, Any], method: str, param: float, out_csv: str):
    # qtable is a flat dict: keys are (state, action), values are Q-values (float)
    # Build per-state action lists and q-values
    per_state: Dict[Any, Dict[Any, float]] = {}
    for key, q in qtable.items():
        if isinstance(key, tuple) and len(key) == 2:
            state, action = key
            per_state.setdefault(state, {})[action] = float(q)
    # Collect global action order
    actions = sorted({a for ad in per_state.values() for a in ad.keys()}, key=lambda a: str(a))

    lines = []
    header = ['state'] + [f"P({str(a)})" for a in actions] + ['best_action']
    lines.append(','.join(header))

    for state, qdict in per_state.items():
        q_values = [qdict.get(a, float('-inf')) for a in actions]
        if method == 'softmax':
            probs = softmax(q_values, temperature=param)
        elif method == 'epsilon_greedy':
            probs = epsilon_greedy(q_values, epsilon=param)
        else:
            raise ValueError('Unknown method: ' + method)
        best_idx = max(range(len(q_values)), key=lambda i: q_values[i])
        best_action = str(actions[best_idx])
        state_str = repr(state)
        line = [state_str] + [f"{p:.6f}" for p in probs] + [best_action]
        lines.append(','.join(line))

    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Export probabilistic policy table from Q-table')
    parser.add_argument('--qtable', type=str, default='', help='Path to qtable .pkl; if empty, auto-pick latest in logs/**/qtables')
    parser.add_argument('--method', type=str, choices=['softmax', 'epsilon_greedy'], default='softmax')
    parser.add_argument('--param', type=float, default=0.5, help='temperature for softmax or epsilon for epsilon_greedy')
    parser.add_argument('--out', type=str, default='policy_table.csv', help='Output CSV path')
    args = parser.parse_args()

    qtable_path = args.qtable
    if not qtable_path:
        candidates = []
        for pattern in [
            os.path.join('logs', '**', 'qtables', '*.pkl'),
            os.path.join('**', 'qtables', '*.pkl'),
        ]:
            candidates += glob.glob(pattern, recursive=True)
        if not candidates:
            raise FileNotFoundError('No qtable .pkl found under logs/**/qtables')
        qtable_path = max(candidates, key=os.path.getmtime)

    qtable = load_qtable(qtable_path)
    export_policy(qtable, method=args.method, param=args.param, out_csv=args.out)
    print(f'Exported policy to {args.out} (method={args.method}, param={args.param}, qtable={qtable_path})')


if __name__ == '__main__':
    main()
