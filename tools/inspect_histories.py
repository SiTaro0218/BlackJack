import glob, os, sys
files = glob.glob(os.path.join('logs','*.history.csv'))
print('Found', len(files), 'history files; showing up to 10:')
for f in files[:10]:
    print(' -', f)
if not files:
    sys.exit(0)
sample = files[0]
print('\n--- sample header/content (first 20 lines):', sample)
with open(sample, 'r', encoding='utf-8', errors='replace') as fh:
    for i,line in enumerate(fh):
        print(line.rstrip('\n'))
        if i>=19: break
