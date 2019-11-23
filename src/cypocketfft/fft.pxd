# cython: language_level=3

from libc.stdint cimport *
from cypocketfft.wrapper cimport cfft_plan, rfft_plan


ctypedef double complex complex_t

cdef size_t _rfft_length(double[:] in_arr) nogil except -1
cdef size_t _irfft_length(complex_t[:] in_arr) nogil except -1

cdef Py_ssize_t _rfft(double[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _rfft_with_plan(rfft_plan* plan, double[:] in_arr, complex_t[:] out_arr, double fct) nogil except -1
cdef Py_ssize_t _irfft(complex_t[:] in_arr, double[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _irfft_with_plan(rfft_plan* plan, complex_t[:] in_arr, double[:] out_arr, double fct) nogil except -1

cdef Py_ssize_t _cfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _icfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _cfft_execute(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=*, bint use_cache=*) nogil except -1
cdef Py_ssize_t _cfft_with_plan(cfft_plan* plan, complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=*) nogil except -1
