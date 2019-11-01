#! /usr/bin/env python

from pathlib import Path
import subprocess
import shlex

TEST_DIR = Path(__file__).resolve().parent / 'tests'

def run():
    pyx_files = ' '.join([str(p) for p in TEST_DIR.glob('*.pyx')])
    cmd_str = 'cythonize -b -a -i {}'.format(pyx_files)
    print(cmd_str)
    subprocess.check_call(shlex.split(cmd_str))

if __name__ == '__main__':
    run()
