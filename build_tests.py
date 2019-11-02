#! /usr/bin/env python

import os
from pathlib import Path
import subprocess
import shlex

class ChDir(object):
    def __init__(self, cur_dir):
        self.orig_dir = None
        self.cur_dir = cur_dir
    def __enter__(self):
        self.orig_dir = Path.cwd()
        os.chdir(self.cur_dir)
        return self
    def __exit__(self, *args):
        os.chdir(self.orig_dir)

TEST_DIR = Path(__file__).resolve().parent / 'tests'

def run():
    cmd_str = 'python setup.py build_ext --inplace'
    print(cmd_str)
    with ChDir(TEST_DIR):
        assert Path.cwd() == TEST_DIR
        subprocess.check_call(shlex.split(cmd_str))

if __name__ == '__main__':
    run()
