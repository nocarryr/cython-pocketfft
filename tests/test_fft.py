import pytest

import numpy as np

from cypocketfft import fft
import _testfft


def dft(x):
    """Discrete Fourier Transform to test against

    Taken from https://github.com/numpy/numpy/blob/b2b45fe9b65db490b84e90fea587c5bb867906eb/numpy/fft/tests/test_pocketfft.py#L17-L21

    Warning:
        This is obviously slow and inefficient! Do not use for large inputs
    """
    L = len(x)
    phase = -2j*np.pi*(np.arange(L)/L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x*np.exp(phase), axis=1)


def build_signal(*shape):
    """Build an array with given shape filled with random values of float64
    """
    N = np.prod(shape)
    sig = np.random.uniform(-1, 1, N)
    return np.reshape(sig, shape)

def build_signal_complex(*shape):
    """Build an array with given shape filled with random values of complex128
    """
    return build_signal(*shape) + 1j*build_signal(*shape)


def test_rfft():
    nreps = 8
    nfft = 1024
    N = nreps * nfft

    sig = build_signal(nreps, nfft)
    for i in range(nreps):
        ff = fft.rfft(sig[i])
        assert np.allclose(ff, dft(sig[i])[:(nfft//2 + 1)])
        iff = fft.irfft(ff)
        assert np.allclose(iff, sig[i])

        npff = np.fft.rfft(sig[i])
        npiff = np.fft.irfft(npff)

        assert np.allclose(ff, npff)
        assert np.allclose(iff, npiff)

@pytest.mark.parametrize('use_omp', [False, True])
def test_rfft_cdef(fft_length, use_omp):
    nreps = 64
    sig = build_signal(nreps, fft_length)
    cy_dur, np_dur = _testfft.test_rfft(sig, use_omp=use_omp, chunksize=4)
    if cy_dur > np_dur:
        diff = (cy_dur - np_dur) * 1000
        print('!!! cy > np: fft_length={}, use_omp={}, diff={} ms'.format(
            fft_length, use_omp, diff,
        ))
    if fft_length > 1024:
        assert cy_dur < np_dur


def test_cfft():
    nfft = 1024
    nreps = 8
    N = nfft * nreps

    sig = build_signal_complex(nreps, nfft)

    for i in range(nreps):
        ff = fft.fft(sig[i])
        assert np.allclose(dft(sig[i]), ff)
        iff = fft.ifft(ff)
        assert np.allclose(sig[i], iff)

        npff = np.fft.fft(sig[i])
        npiff = np.fft.ifft(npff)

        assert np.allclose(ff, npff)
        assert np.allclose(iff, npiff)

@pytest.mark.parametrize('use_omp', [False, True])
def test_cfft_cdef(fft_length, use_omp):
    _testfft.test_cfft(nfft=fft_length, reps=8, use_omp=use_omp)
