# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
import time
import cProfile
import pstats

from cypocketfft.wrapper cimport *
from cypocketfft import fft
from cypocketfft cimport fft
from cypocketfft.fft cimport REAL_ft, COMPLEX_ft

@cython.boundscheck(False)
@cython.wraparound(False)
def test_native(Py_ssize_t N=32, Py_ssize_t reps=1, object pr=None):
    t = np.linspace(0., 1., N)
    fc = 6000.
    x = np.sin(2*np.pi*fc*t)
    cdef double[:] x_view = x
    ff_length = fft._rfft_length(x_view)
    ff = np.empty((reps, ff_length), dtype=np.complex128)
    cdef COMPLEX_ft[:,:] ff_view = ff
    cdef double[:,:] iff_view = np.empty((reps, N), dtype=np.float64)
    cdef Py_ssize_t i
    cdef double iff_coef = 1. / <double>N

    if pr is None:
        pr = cProfile.Profile()

    pr.enable()
    for i in range(reps):
        fft._rfft(x_view, ff_view[i,:], 1.0)
    for i in range(reps):
        fft._irfft(ff_view[i,:], iff_view[i,:], iff_coef)
    pr.disable()

    return pr

@cython.boundscheck(False)
@cython.wraparound(False)
def test_numpy(Py_ssize_t N=32, Py_ssize_t reps=1, object pr=None):
    t = np.linspace(0., 1., N)
    fc = 6000.
    x = np.sin(2*np.pi*fc*t)
    ff_length = fft._rfft_length(x)
    cdef COMPLEX_ft[:,:] npff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef COMPLEX_ft[:] npff
    cdef double[:,:] npiff_view = np.empty((reps, N), dtype=np.float64)
    cdef double[:] npiff
    cdef Py_ssize_t i

    if pr is None:
        pr = cProfile.Profile()

    pr.enable()
    for i in range(reps):
        npff = np.fft.rfft(x)
        npff_view[i,:] = npff
    for i in range(reps):
        npiff = np.fft.irfft(npff_view[i,:])
        npiff_view[i,:] = npiff
    pr.disable()

    return pr

def test(Py_ssize_t N=32, Py_ssize_t reps=1, return_result=False):
    t = np.linspace(0., 1., N)
    fc = 6000.
    x = np.sin(2*np.pi*fc*t)
    cdef double[:] x_view = x
    ff_length = fft._rfft_length(x_view)
    # ff = np.empty(ff_length, dtype=np.complex128)
    ff = np.empty((reps, ff_length), dtype=np.complex128)
    cdef COMPLEX_ft[:,:] ff_view = ff

    # fft._rfft(x_view, ff_view[0,:], 1.0)
    start_ts = time.time()

    cdef Py_ssize_t i
    # with nogil:
    #     for i in prange(reps):
    #         fft._rfft(x_view, ff_view[i,:], 1.0)

    with nogil:
        for i in prange(reps):
            fft._rfft(x_view, ff_view[i,:], 1.0)

    end_ts = time.time()
    dur = end_ts - start_ts
    percall = dur / float(reps)
    print('cython duration={}, percall={}'.format(dur, percall))

    # npff = np.empty((reps, ff_length), dtype=np.complex128)
    cdef COMPLEX_ft[:,:] npff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef COMPLEX_ft[:] npff# = np.empty(ff_length, dtype=np.complex128)

    np_start_ts = time.time()
    for i in range(reps):
        npff = np.fft.rfft(x)
        npff_view[i,:] = npff
    np_end_ts = time.time()
    np_dur = np_end_ts - np_start_ts
    np_percall = np_dur / float(reps)
    print('numpy duration={}, percall={}'.format(np_dur, np_percall))

    ff = ff[0]
    assert np.allclose(ff_view, npff_view)

    iff = fft.irfft(ff_view[0,:])
    npiff = np.fft.irfft(ff_view[0,:])
    assert np.allclose(iff, npiff)
    assert np.allclose(iff, x)

    if return_result:
        return x, ff, iff
