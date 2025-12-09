import os, glob, time
BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
print('Project base:', BASE)
os.chdir(BASE)

# list qtables
qt = sorted(glob.glob('logs/qtables/*.pkl'))
print('\nQ-table count:', len(qt))
for p in qt[:20]:
    print(' -', p)
if len(qt) > 20:
    print(' ... and', len(qt)-20, 'more')

# recent log files
logs = sorted(glob.glob('logs/*.txt') + glob.glob('logs/*.history.csv'), key=os.path.getmtime, reverse=True)
print('\nRecent logs (up to 30):')
for p in logs[:30]:
    m = time.ctime(os.path.getmtime(p))
    print(m, '-', p)

# results.csv
if os.path.exists('results.csv'):
    print('\nresults.csv exists; head:')
    with open('results.csv', 'r', encoding='utf-8', errors='replace') as f:
        for i,line in enumerate(f):
            print(line.strip())
            if i >= 9:
                break
else:
    print('\nresults.csv not found')

# check smoke / recent result files created by run_experiments
for fname in ['smoke_results.csv', 'results.csv']:
    if os.path.exists(fname):
        print('\nFile', fname, 'size', os.path.getsize(fname))

# check for any traceback in recent txt logs
print('\nSearch for "Traceback" in recent .txt logs:')
for p in logs[:30]:
    if p.endswith('.txt'):
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                txt = f.read()
            if 'Traceback' in txt or 'Error' in txt:
                print('---', p, 'contains error-like text (showing first 200 chars):')
                print(txt[:200].replace('\n','\\n'))
        except Exception as e:
            print('Could not read', p, e)

print('\nCheck complete')
