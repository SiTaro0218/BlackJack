#!/usr/bin/env python3
"""Merge an autocollect results CSV into main results.csv, clean duplicates, then run visualization.

Usage: python merge_and_visualize.py <autocollect_csv>
"""
import sys
import os
import csv
import subprocess


def append_csv(src, dst='results.csv'):
    if not os.path.exists(src):
        print('Source file not found:', src)
        return 1

    # If dst doesn't exist, just copy src to dst
    if not os.path.exists(dst):
        # simply copy contents
        with open(src, newline='', encoding='utf-8') as fsrc, open(dst, 'w', newline='', encoding='utf-8') as fdst:
            fdst.write(fsrc.read())
        print(f'Created {dst} from {src}')
        return 0

    # Append rows from src (skip header) to dst
    with open(src, newline='', encoding='utf-8') as fsrc, open(dst, 'a', newline='', encoding='utf-8') as fdst:
        reader = csv.reader(fsrc)
        writer = csv.writer(fdst)
        header = next(reader, None)
        if header is None:
            print('No data in', src)
            return 1
        for row in reader:
            writer.writerow(row)
    print(f'Appended rows from {src} into {dst}')
    return 0


def main():
    if len(sys.argv) < 2:
        print('Usage: python merge_and_visualize.py <autocollect_csv>')
        sys.exit(2)

    src = sys.argv[1]
    # defensive: strip quotes
    src = src.strip('"')

    rc = append_csv(src, dst='results.csv')
    if rc != 0:
        print('Append failed')
        sys.exit(rc)

    # run clean_results.py to dedupe
    print('Running clean_results.py to dedupe results.csv...')
    try:
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'clean_results.py')])
    except subprocess.CalledProcessError as e:
        print('clean_results.py failed with', e)
        sys.exit(3)

    # run visualization
    print('Running visualize_results.py...')
    try:
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'visualize_results.py')])
    except subprocess.CalledProcessError as e:
        print('visualize_results.py failed with', e)
        sys.exit(4)

    print('Merge + clean + visualize completed')


if __name__ == '__main__':
    main()
