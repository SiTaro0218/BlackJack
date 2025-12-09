import os
from collections import defaultdict

QT_DIR = os.path.join('logs','autocollect_FULL_20251110_121110','qtables')
files = [f for f in os.listdir(QT_DIR) if f.endswith('.pkl')]
params = defaultdict(set)
seeds = set()
for fn in files:
    # example: alpha-0_05--gamma-0_98--eps_start-1_0--eps_end-0_01--eps_decay_type-linear--bins-medium--seed-3.pkl
    parts = fn.replace('.pkl','').split('--')
    for p in parts:
        if p.startswith('alpha-'):
            params['alpha'].add(p.split('alpha-')[1].replace('_','.'))
        elif p.startswith('gamma-'):
            params['gamma'].add(p.split('gamma-')[1].replace('_','.'))
        elif p.startswith('eps_start-'):
            params['eps_start'].add(p.split('eps_start-')[1].replace('_','.'))
        elif p.startswith('eps_end-'):
            params['eps_end'].add(p.split('eps_end-')[1].replace('_','.'))
        elif p.startswith('eps_decay_type-'):
            params['eps_decay_type'].add(p.split('eps_decay_type-')[1])
        elif p.startswith('bins-'):
            params['bins'].add(p.split('bins-')[1])
        elif p.startswith('seed-'):
            seeds.add(p.split('seed-')[1])

# sort values
for k in params:
    params[k] = sorted(params[k], key=lambda x: float(x) if all(c.isdigit() or c=='.' for c in x) else x)
seeds = sorted(seeds, key=lambda x: int(x))

# Build LaTeX section
lines = []
lines.append('\n\\section*{実験で試したハイパーパラメータ}\n')
lines.append('以下のパラメータ空間でグリッドサーチ的に実験を実行した．\\n')
lines.append('\\begin{itemize}\n')
lines.append(f"  \\item 学習率 $\\alpha$: {', '.join(params['alpha'])}\\n")
lines.append(f"  \\item 割引率 $\\gamma$: {', '.join(params['gamma'])}\\n")
lines.append(f"  \\item 盤分割（bins）プリセット: {', '.join(params['bins'])}\\n")
lines.append(f"  \\item $\\epsilon$ 開始値（eps\_start）: {', '.join(params['eps_start'])}\\n")
lines.append(f"  \\item $\\epsilon$ 終了値（eps\_end）: {', '.join(params['eps_end'])}\\n")
lines.append(f"  \\item $\\epsilon$ 減衰タイプ: {', '.join(params['eps_decay_type'])}\\n")
lines.append(f"  \\item 乱数シード: {', '.join(seeds)}\\n")
lines.append('  \\item 学習エピソード数: 1000（デフォルト，1 run 当たり）\\n')
lines.append('  \\item 最大ステップ数/エピソード: 500\\n')
lines.append('  \\item 各組合せの試行回数: seeds に示す数（ログにより保存）\\n')
lines.append('\\end{itemize}\n')

print('\n'.join(lines))
