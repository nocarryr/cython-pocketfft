# cython: language_level=3

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
from cypocketfft.fft cimport complex_t

cdef double get_time() nogil except *:
    cdef double result
    IF UNAME_SYSNAME == "Windows":
        with gil:
            result = time.time()
    ELSE:
        cdef timespec t
        clock_gettime(CLOCK_REALTIME, &t)
        result = <double>t.tv_sec
        result += <double>t.tv_nsec / <double>1000000000
    return result

cdef void _handle_rfft(double[:] time_domain, complex_t[:] freq_domain, double fct, bint is_forward) nogil except *:
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
    cdef complex_t[:] ff = np.empty(ff_length, dtype=np.complex128)
    cdef complex_t[:,:] ff_view = np.empty((reps, ff_length), dtype=np.complex128)
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
    cdef complex_t[:,:] npff_view = np.empty((reps, ff_length), dtype=np.complex128)
    cdef complex_t[:] npff
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
def test_rfft(double[:,:] sig, bint use_omp=False, Py_ssize_t chunksize=1):
    cdef Py_ssize_t nreps = sig.shape[0], in_length = sig.shape[1]
    cdef Py_ssize_t ff_length = fft._rfft_length(sig[0])
    cdef complex_t[:,:] ff_view = np.empty((nreps, ff_length), dtype=np.complex128)
    cdef double[:,:] iff_view = np.empty((nreps, in_length), dtype=np.float64)
    cdef double i_fct = 1.0 / <double>in_length

    cdef Py_ssize_t i
    cdef double start_ts, end_ts

    if use_omp:
        with nogil:
            start_ts = get_time()
            for i in prange(nreps):
                fft._rfft(sig[i,:], ff_view[i,:], 1.0)
                fft._irfft(ff_view[i,:], iff_view[i,:], i_fct)
    else:
        start_ts = get_time()
        with nogil:
            for i in range(nreps):
                fft._rfft(sig[i], ff_view[i], 1.0)
                fft._irfft(ff_view[i], iff_view[i], i_fct)

    end_ts = get_time()
    cy_dur = end_ts - start_ts
    print('')
    print('length={}, use_omp={}'.format(in_length, use_omp))
    print('cython duration={}'.format(cy_dur))

    cdef complex_t[:,:] npff_view = np.empty((nreps, ff_length), dtype=np.complex128)
    cdef complex_t[:] npff = np.empty(ff_length, dtype=np.complex128)
    cdef double[:,:] npiff_view = np.empty((nreps, in_length), dtype=np.float64)
    cdef double[:] npiff = np.empty(in_length, dtype=np.float64)
    cdef double np_start_ts, np_end_ts

    np_start_ts = get_time()
    for i in range(nreps):
        npff = np.fft.rfft(sig[i])
        npff_view[i,:] = npff
        npiff = np.fft.irfft(npff_view[i])
        npiff_view[i,:] = npiff

    np_end_ts = get_time()

    np_dur = np_end_ts - np_start_ts
    print('numpy duration={}'.format(np_dur))

    assert np.allclose(ff_view, npff_view)
    assert np.allclose(iff_view, npiff_view)
    return cy_dur, np_dur

@cython.boundscheck(False)
@cython.wraparound(False)
def test_cfft(Py_ssize_t nfft=32, Py_ssize_t reps=1, bint use_omp=False):
    cdef Py_ssize_t N = nfft * reps
    cdef double i_fct = 1.0 / <double>nfft

    _sig = np.random.uniform(-1, 1, N) + 1j*np.random.uniform(-1, 1, N)
    cdef complex_t[:,:] sig = np.reshape(_sig, (reps, nfft))
    cdef complex_t[:,:] ff_view = np.empty((reps, nfft), dtype=np.complex128)
    cdef complex_t[:,:] iff_view = np.empty((reps, nfft), dtype=np.complex128)

    cdef Py_ssize_t i
    cdef double start_ts, end_ts

    if use_omp:
        start_ts = get_time()
        with nogil:
            for i in prange(reps):
                fft._cfft(sig[i], ff_view[i,:], 1.0)
                fft._icfft(ff_view[i], iff_view[i], i_fct)
    else:
        start_ts = get_time()
        with nogil:
            for i in range(reps):
                fft._cfft(sig[i], ff_view[i,:], 1.0)
                fft._icfft(ff_view[i], iff_view[i], i_fct)

    end_ts = get_time()
    dur = end_ts - start_ts
    percall = dur / <double>reps
    print('cython duration={}, percall={}'.format(dur, percall))

    cdef complex_t[:,:] npff_view = np.empty((reps, nfft), dtype=np.complex128)
    cdef complex_t[:,:] npiff_view = np.empty((reps, nfft), dtype=np.complex128)
    cdef complex_t[:] npff
    cdef complex_t[:] npiff
    cdef double np_start_ts, np_end_ts

    np_start_ts = get_time()
    for i in range(reps):
        npff = np.fft.fft(sig[i])
        npff_view[i,:] = npff
        npiff = np.fft.ifft(npff)
        npiff_view[i,:] = npiff
    np_end_ts = get_time()

    np_dur = np_end_ts - np_start_ts
    np_percall = np_dur / <double>reps
    print('numpy duration={}, percall={}'.format(np_dur, np_percall))

    assert np.allclose(ff_view, npff_view)
    assert np.allclose(iff_view, npiff_view)
    assert np.allclose(sig, iff_view)
