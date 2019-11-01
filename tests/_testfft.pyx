# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# distutils: include_dirs = ../cypocketfft/_pocketfft_lib

cimport cython
from cython.parallel cimport prange
IF UNAME_SYSNAME != "Windows":
    from posix.time cimport timespec, clock_gettime, CLOCK_REALTIME
import numpy as np
# cimport numpy as np
import time
import cProfile
import pstats

from cypocketfft.wrapper cimport *
from cypocketfft import fft
from cypocketfft cimport fft
from cypocketfft.fft cimport REAL_ft, COMPLEX_ft

cdef double get_time() nogil except *:
    cdef double result
    IF UNAME_SYSNAME == "Windows":
        result = time.time()
    ELSE:
        cdef timespec t
        clock_gettime(CLOCK_REALTIME, &t)
        result = <double>t.tv_sec
        result += <double>t.tv_nsec / <double>1000000000
    return result

cdef void _handle_rfft(double[:] time_domain, COMPLEX_ft[:] freq_domain, double fct, bint is_forward) nogil except *:
    if is_forward:
        fft._rfft(time_domain, freq_domain, fct)
    else:
        fft._irfft(freq_domain, time_domain, fct)

@cython.boundscheck(False)
@cython.wraparound(False)
def test_native(Py_ssize_t N=32, Py_ssize_t reps=1, object pr=None):
    t = np.linspace(0., 1., N)
    fc = 6000.
    x = np.sin(2*np.pi*fc*t)
    cdef double[:] x_view = x
    ff_length = fft._rfft_length(x_view)
    cdef double complex[:] ff = np.empty(ff_length, dtype=np.complex128)
    cdef double complex[:,:] ff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef double[:] iff = np.empty(N, dtype=np.float64)
    cdef double[:,:] iff_view = np.empty((reps, N), dtype=np.float64)
    cdef Py_ssize_t i
    cdef double iff_coef = 1. / <double>N

    if pr is None:
        pr = cProfile.Profile()

    pr.enable()
    for i in range(reps):
        _handle_rfft(x_view, ff, 1.0, True)
        ff_view[i,:] = ff
    for i in range(reps):
        _handle_rfft(iff, ff_view[i,:], iff_coef, False)
        iff_view[i,:] = iff
    pr.disable()

    return pr

@cython.boundscheck(False)
@cython.wraparound(False)
def test_numpy(Py_ssize_t N=32, Py_ssize_t reps=1, object pr=None):
    t = np.linspace(0., 1., N)
    fc = 6000.
    x = np.sin(2*np.pi*fc*t)
    cdef double[:] x_view = x
    ff_length = fft._rfft_length(x_view)
    cdef double complex[:,:] npff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef double complex[:] npff
    cdef double[:,:] npiff_view = np.empty((reps, N), dtype=np.float64)
    cdef double[:] npiff = np.empty(N, dtype=np.float64)
    cdef Py_ssize_t i

    if pr is None:
        pr = cProfile.Profile()

    pr.enable()
    for i in range(reps):
        npff = np.fft.rfft(x_view)
        npff_view[i,:] = npff
    for i in range(reps):
        npiff = np.fft.irfft(npff_view[i,:])
        npiff_view[i,:] = npiff
    pr.disable()

    return pr

@cython.boundscheck(False)
@cython.wraparound(False)
def test(Py_ssize_t N=32, Py_ssize_t reps=1, bint return_result=False, bint use_omp=False):
    t = np.linspace(0., 1., N)
    fc = 6000.
    cdef double[:] x = np.sin(2*np.pi*fc*t)
    cdef double[:] x_view = x
    cdef Py_ssize_t ff_length = fft._rfft_length(x_view)
    cdef double complex[:,:] ff_view = np.empty((reps, ff_length), dtype=np.complex128)

    cdef Py_ssize_t i
    cdef double start_ts, end_ts

    if use_omp:
        start_ts = get_time()
        with nogil:
            for i in prange(reps):
                fft._rfft(x_view, ff_view[i,:], 1.0)
    else:
        start_ts = get_time()
        with nogil:
            for i in range(reps):
                fft._rfft(x_view, ff_view[i,:], 1.0)

    end_ts = get_time()
    dur = end_ts - start_ts
    percall = dur / <double>reps
    print('cython duration={}, percall={}'.format(dur, percall))

    cdef double complex[:,:] npff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef double complex[:] npff
    cdef double np_start_ts, np_end_ts

    np_start_ts = get_time()
    for i in range(reps):
        npff = np.fft.rfft(x)
        npff_view[i,:] = npff
    np_end_ts = get_time()

    np_dur = np_end_ts - np_start_ts
    np_percall = np_dur / <double>reps
    print('numpy duration={}, percall={}'.format(np_dur, np_percall))

    assert np.allclose(ff_view, npff_view)

    cdef double[:] iff = fft.irfft(ff_view[0,:])
    cdef double[:] npiff = np.fft.irfft(ff_view[0,:])
    assert np.allclose(iff, npiff)
    assert np.allclose(iff, x)

    if return_result:
        return np.asarray(x), np.asarray(ff_view[0]), np.asarray(iff)
