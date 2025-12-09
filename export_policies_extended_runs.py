import pickle
import csv
from pathlib import Path
from classes import Action

def load_qtable(path: Path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'table' in obj:
        table = obj['table']
    else:
        table = obj
    # table is mapping of (state, Action)->Q_value
    # Reconstruct state-> {Action:Q}
    state_map = {}
    for key, q in table.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        state, action = key
        if state not in state_map:
            state_map[state] = {}
        state_map[state][action] = q
    return state_map

def action_list():
    return [Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER, Action.RETRY]

def greedy_policy(state_map):
    policy = {}
    for state, qvals in state_map.items():
        best_a = max(qvals.items(), key=lambda kv: kv[1])[0]
        policy[state] = best_a
    return policy

def softmax_policy(state_map, temperature: float = 1.0):
    import math
    pol = {}
    for state, qvals in state_map.items():
        acts = list(qvals.keys())
        vals = [qvals[a] for a in acts]
        max_v = max(vals)
        exps = [math.exp((v - max_v)/temperature) for v in vals]
        s = sum(exps)
        probs = [e/s for e in exps]
        pol[state] = list(zip(acts, probs))
    return pol

def write_greedy(policy, path: Path):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['score','hand_length','retry_bucket','greedy_action'])
        for (score,l,r), act in sorted(policy.items()):
            w.writerow([score,l,r, act.name.lower()])

def write_softmax(pol, path: Path):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['score','hand_length','retry_bucket'] + [a.name.lower() for a in action_list()]
        w.writerow(header)
        for (score,l,r), dist in sorted(pol.items()):
            prob_map = {a.name.lower():0.0 for a in action_list()}
            for a, p in dist:
                prob_map[a.name.lower()] = p
            w.writerow([score,l,r] + [prob_map[a.name.lower()] for a in action_list()])

def main():
    base = Path('logs/extended_runs')
    runs = [
        'rep_a015_g095_ps05_10k',
        'rep_a015_g095_ps03_10k',
        'rep_a015_g095_ps07_10k',
    ]
    for run in runs:
        qpath = base / run / 'qtable.pkl'
        if not qpath.exists():
            print(f'[WARN] missing qtable for {run}')
            continue
        state_map = load_qtable(qpath)
        gp = greedy_policy(state_map)
        sp = softmax_policy(state_map, temperature=1.0)
        write_greedy(gp, base / run / 'policy_greedy.csv')
        write_softmax(sp, base / run / 'policy_softmax_t1.csv')
        print(f'Exported policies for {run}')

if __name__ == '__main__':
    main()
