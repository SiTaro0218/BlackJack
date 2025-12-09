import glob, os
for s in [1,2,3,4]:
    print('\n=== Seed', s, '===')
    files = sorted(glob.glob(f'logs/*seed-{s}.txt'))
    if not files:
        print(' no .txt logs for seed', s)
        continue
    for p in files:
        print('FILE:', p)
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                txt = f.read()
            if 'Traceback' in txt or 'Exception' in txt or 'Error' in txt or 'Traceback (most recent call last):' in txt:
                # print first 500 chars
                i = txt.find('Traceback')
                if i==-1:
                    i = 0
                print('--- snippet ---')
                print(txt[i:i+500].replace('\n','\\n'))
            else:
                print(' No obvious exception/traceback text found in this file (first 200 chars):')
                print(txt[:200].replace('\n','\\n'))
        except Exception as e:
            print('Could not read', p, e)
