import os, csv, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
# allow overriding paths via CLI
hist = os.path.join(BASE, 'logs', 'test_top1_long.history.csv')
out = os.path.join(BASE, 'figures', 'test_top1_long_money.png')
if len(sys.argv) > 1:
    hist = sys.argv[1]
if len(sys.argv) > 2:
    out = sys.argv[2]
# ensure config importable
sys.path.insert(0, BASE)
try:
    from config import INITIAL_MONEY
except Exception:
    INITIAL_MONEY = 10000

if not os.path.exists(hist):
    print('History file not found:', hist)
    raise SystemExit(2)

# parse per-episode rewards
rewards = []
cur_sum = 0.0
with open(hist, newline='', encoding='utf-8', errors='replace') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            rew = float(row.get('reward', 0) or 0)
        except:
            rew = 0.0
        cur_sum += rew
        res = (row.get('result') or '').strip().lower()
        if res and res not in ['', 'unsettled', 'retry']:
            rewards.append(cur_sum)
            cur_sum = 0.0
if cur_sum != 0.0:
    rewards.append(cur_sum)

print('Parsed', len(rewards), 'episodes from', hist)
start_money = INITIAL_MONEY
money = [start_money]
for r in rewards:
    money.append(money[-1] + r)

final_money = money[-1]
delta = final_money - start_money
print('Start money:', start_money)
print('Final money:', final_money)
print('Delta:', delta)

plt.figure(figsize=(12,6))
plt.plot(range(0, len(money)), money, marker=None)
plt.xlabel('Episode')
plt.ylabel('Money')
plt.title(f'Money progression (long test)  start {start_money} -> final {final_money} (Δ {delta})')
plt.grid(True)
plt.tight_layout()
plt.savefig(out)
print('Saved figure to', out)
