import pytest

import _test_plancache

def test():
    _test_plancache.test()

def test_concurrency():
    nthreads = 4
    nplans = 32
    _test_plancache.test_concurrency(nthreads, nplans)
