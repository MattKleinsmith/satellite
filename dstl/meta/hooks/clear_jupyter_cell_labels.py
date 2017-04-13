#!/usr/bin/env python3
import argparse
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('f', help='.py file generated from a .ipynb')
args = parser.parse_args()
tmp = args.f + '.tmp'
with open(args.f) as fin, open(tmp, 'w') as fout:
    for line in fin:
        if not re.match('# In\[[0-9]*\]:\n', line):
            print(line, end='', file=fout)
shutil.move(tmp, args.f)
