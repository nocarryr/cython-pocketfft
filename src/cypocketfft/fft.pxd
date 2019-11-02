# cython: language_level=3
# distutils: include_dirs = src/cypocketfft/_pocketfft_lib
# distutils: sources = src/cypocketfft/_pocketfft_lib/pocketfft.c

from libc.stdint cimport *
from cypocketfft.wrapper cimport cfft_plan, rfft_plan


ctypedef fused REAL_ft:
    # float
    double

ctypedef fused COMPLEX_ft:
    # float complex
    double complex

cdef size_t _rfft_length(REAL_ft[:] in_arr) nogil except -1
cdef size_t _irfft_length(COMPLEX_ft[:] in_arr) nogil except -1

cdef Py_ssize_t _rfft(REAL_ft[:] in_arr, COMPLEX_ft[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _rfft_with_plan(rfft_plan* plan, REAL_ft[:] in_arr, COMPLEX_ft[:] out_arr, double fct) nogil except -1
cdef Py_ssize_t _irfft(COMPLEX_ft[:] in_arr, REAL_ft[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _irfft_with_plan(rfft_plan* plan, COMPLEX_ft[:] in_arr, REAL_ft[:] out_arr, double fct) nogil except -1
