# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cimport cython

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from cypocketfft cimport wrapper
from cypocketfft.plancache cimport plan_cache

import numpy as np

cdef size_t _rfft_length(double[:] in_arr) nogil except -1:
    cdef size_t in_size = in_arr.shape[0]
    cdef size_t length = in_size // 2 + 1
    return length

def rfft_length(double[:] in_arr):
    """Calculate the rfft result length for the input array

    This evaluates to :math:`N / 2 + 1`
    """
    return _rfft_length(in_arr)


cdef size_t _irfft_length(complex_t[:] in_arr) nogil except -1:
    cdef size_t in_size = in_arr.shape[0]
    cdef size_t length = (in_size-1) * 2
    return length

def irfft_length(complex_t[:] in_arr):
    """Calculate the irfft result length for the input array

    This evaluates to :math:`(N - 1) * 2`
    """
    return _irfft_length(in_arr)


cdef Py_ssize_t _rfft(double[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True) nogil except -1:
    cdef Py_ssize_t in_size = in_arr.shape[0]
    cdef rfft_plan plan
    if use_cache:
        plan = plan_cache.get_rplan(in_size)
    else:
        plan = plan_cache._build_rplan(in_size)
    cdef Py_ssize_t r = _rfft_with_plan(&plan, in_arr, out_arr, fct)
    if not use_cache and plan != NULL:
        wrapper._destroy_rfft_plan(plan)
    return r

cdef Py_ssize_t _rfft_with_plan(rfft_plan* plan, double[:] in_arr, complex_t[:] out_arr, double fct) nogil except -1:
    cdef Py_ssize_t in_size = in_arr.shape[0]
    cdef Py_ssize_t out_size = _rfft_length(in_arr)
    if out_arr.shape[0] < out_size:
        with gil:
            raise Exception('out_arr shape mismatch. size={}, expected={}'.format(
                out_arr.shape[0], out_size,
            ))

    cdef void *in_ptr = &in_arr[0]
    cdef void *out_ptr = &out_arr[0]

    out_arr[out_size-1].imag = 0
    memcpy(<char *>out_ptr+sizeof(double), in_ptr, in_size*sizeof(double))

    cdef double *out_ptr_dbl = <double *>out_ptr
    r = wrapper._rfft_forward(plan[0], out_ptr_dbl+1, fct)
    if r == 0:
        out_arr[0].real = out_arr[0].imag
        out_arr[0].imag = 0
    if r != 0:
        with gil:
            raise MemoryError()
    return out_size

cdef Py_ssize_t _irfft(complex_t[:] in_arr, double[:] out_arr, double fct, bint use_cache=True) nogil except -1:
    cdef Py_ssize_t in_size = in_arr.shape[0]
    cdef Py_ssize_t length = _irfft_length(in_arr)
    cdef rfft_plan plan
    if use_cache:
        plan = plan_cache.get_rplan(length)
    else:
        plan = plan_cache._build_rplan(length)
    r = _irfft_with_plan(&plan, in_arr, out_arr, fct)
    if not use_cache and plan != NULL:
        wrapper._destroy_rfft_plan(plan)
    return r

cdef Py_ssize_t _irfft_with_plan(rfft_plan* plan, complex_t[:] in_arr, double[:] out_arr, double fct) nogil except -1:
    cdef Py_ssize_t in_size = in_arr.shape[0]
    cdef Py_ssize_t length = _irfft_length(in_arr)
    if out_arr.shape[0] < length:
        with gil:
            raise Exception('out_arr shape mismatch. size={}, expected={}'.format(
                out_arr.shape[0], length,
            ))
    cdef void *in_ptr = &in_arr[0]
    cdef void *out_ptr = &out_arr[0]

    memcpy(<char *>out_ptr+sizeof(double), <char *>in_ptr + (sizeof(double)*2), (length-1)*sizeof(double))
    out_arr[0] = in_arr[0].real

    cdef double *out_ptr_dbl = <double *>out_ptr
    r = wrapper._rfft_backward(plan[0], out_ptr_dbl, fct)
    if r != 0:
        with gil:
            raise MemoryError()
    return length

cdef Py_ssize_t _cfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True) nogil except -1:
    return _cfft_execute(in_arr, out_arr, fct, True, use_cache)

cdef Py_ssize_t _icfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True) nogil except -1:
    return _cfft_execute(in_arr, out_arr, fct, False, use_cache)

cdef Py_ssize_t _cfft_execute(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=True, bint use_cache=True) nogil except -1:
    cdef Py_ssize_t length = in_arr.shape[0]
    cdef cfft_plan plan
    if use_cache:
        plan = plan_cache.get_cplan(length)
    else:
        plan = plan_cache._build_cplan(length)
    if plan is NULL:
        with gil:
            raise Exception()
    cdef Py_ssize_t r = _cfft_with_plan(&plan, in_arr, out_arr, fct, is_forward)
    if not use_cache and plan != NULL:
        wrapper._destroy_cfft_plan(plan)
    return r

cdef Py_ssize_t _cfft_with_plan(cfft_plan* plan, complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=True) nogil except -1:
    cdef Py_ssize_t length = in_arr.shape[0]
    if out_arr.shape[0] != length:
        with gil:
            raise Exception('out_arr shape mismatch. size={}, expected={}'.format(
                out_arr.shape[0], length,
            ))
    cdef void *in_ptr = &in_arr[0]
    cdef void *out_ptr = &out_arr[0]
    memcpy(out_ptr, in_ptr, length * sizeof(complex_t))
    cdef double *out_ptr_dbl = <double *>out_ptr
    if is_forward:
        r = wrapper._cfft_forward(plan[0], out_ptr_dbl, fct)
    else:
        r = wrapper._cfft_backward(plan[0], out_ptr_dbl, fct)
    if r != 0:
        with gil:
            raise MemoryError()
    return r


def rfft(double[:] in_arr, fct=None):
    """Perform the rfft

    Arguments:
        in_arr: Input array or typed-memoryview (real-valued)
        fct: Scaling factor to apply to the un-normalized transform (typically ``1.0``)

    Returns:
        The ifft result as a typed-memoryview of ``complex_t``

    """
    if fct is None:
        fct = 1.0
    cdef double _fct = fct
    out_size = _rfft_length(in_arr)
    cdef complex_t[:] out_arr = np.empty(out_size, dtype=np.complex128)
    _rfft(in_arr, out_arr, _fct)
    assert out_size == out_arr.size
    return out_arr

def irfft(complex_t[:] in_arr, fct=None):
    """Perform the inverse rfft

    Arguments:
        in_arr: Input array or typed-memoryview (complex-valued)
        fct: Scaling factor to apply to the un-normalized transform (typically :math:`1.0 / N`)

    Returns:
        The ifft result as a typed-memoryview of ``double``

    """
    out_size = _irfft_length(in_arr)
    if fct is None:
        fct = <double>(1 / <double>out_size)
    cdef double _fct = fct
    cdef double[:] out_arr = np.empty(out_size, dtype=np.float64)
    _irfft(in_arr, out_arr, _fct)
    return out_arr

def fft(complex_t[:] in_arr, fct=None):
    if fct is None:
        fct = 1.0
    cdef double _fct = fct
    cdef Py_ssize_t length = in_arr.shape[0]
    cdef complex_t[:] out_arr = np.empty(length, dtype=np.complex128)
    _cfft(in_arr, out_arr, _fct)
    return np.asarray(out_arr)

def ifft(complex_t[:] in_arr, fct=None):
    cdef Py_ssize_t length = in_arr.shape[0]
    if fct is None:
        fct = <double>(1 / <double>length)
    cdef double _fct = fct
    cdef complex_t[:] out_arr = np.empty(length, dtype=np.complex128)
    _icfft(in_arr, out_arr, _fct, False)
    return np.asarray(out_arr)
