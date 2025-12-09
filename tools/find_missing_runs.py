import itertools, os
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE, 'logs', 'qtables')

def make_run_name(params):
    parts = []
    for k in ['alpha', 'gamma', 'eps_start', 'eps_end', 'eps_decay_type', 'seed']:
        parts.append(f"{k}={params.get(k)}")
    name = "--".join(parts)
    return name.replace(' ', '').replace('.', '_').replace('/', '_').replace('=', '-')

alphas = [0.01, 0.03, 0.05, 0.1, 0.2]
gammas = [0.9, 0.95, 0.98]
eps_starts = [1.0, 0.5]
eps_ends = [0.01, 0.05]
decay_types = ['const', 'linear', 'exp']
seeds = [0,1,2,3,4]

missing = []
all_runs = []
for a,g,es,ee,dt,s in itertools.product(alphas,gammas,eps_starts,eps_ends,decay_types,seeds):
    params={'alpha':a,'gamma':g,'eps_start':es,'eps_end':ee,'eps_decay_type':dt,'seed':s}
    rn = make_run_name(params)
    p = os.path.join(OUT_DIR, rn + '.pkl')
    all_runs.append((rn,p,params))
    if not os.path.exists(p):
        missing.append((rn,p,params))

print(f'Total expected runs per seed: {len(alphas)*len(gammas)*len(eps_starts)*len(eps_ends)*len(decay_types)}')
print('Missing count:', len(missing))
# print first 5 missing
for rn,p,params in missing[:10]:
    print(rn)

# write missing list
with open(os.path.join(BASE,'logs','missing_runs_seed1-4.txt'), 'w', encoding='utf-8') as f:
    for rn,p,params in missing:
        f.write(rn + '\n')

print('\nWrote logs/missing_runs_seed1-4.txt')
