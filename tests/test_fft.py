import pytest

import _testfft

@pytest.mark.parametrize('use_omp', [False, True])
def test_fft(fft_length, use_omp):
    _testfft.test(N=fft_length, reps=16, use_omp=use_omp)
