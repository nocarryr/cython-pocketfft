# cython: language_level=3

from libc.stdint cimport *
from cypocketfft.wrapper cimport cfft_plan, rfft_plan

ctypedef float float32_t
ctypedef double float64_t

ctypedef struct complex_64_t:
    float real
    float imag

ctypedef struct complex_128_t:
    double real
    double imag

# ctypedef fused REAL_ft:
#     float32_t
#     float64_t
ctypedef float64_t REAL_ft

# ctypedef fused COMPLEX_ft:
#     complex_64_t
#     complex_128_t
ctypedef complex_128_t COMPLEX_ft

cdef Py_ssize_t _rfft(REAL_ft[:] in_arr, COMPLEX_ft[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _rfft_with_plan(rfft_plan* plan, REAL_ft[:] in_arr, COMPLEX_ft[:] out_arr, double fct) nogil except -1
cdef Py_ssize_t _irfft(COMPLEX_ft[:] in_arr, REAL_ft[:] out_arr, double fct, bint use_cache=*) nogil except -1
cdef Py_ssize_t _irfft_with_plan(rfft_plan* plan, COMPLEX_ft[:] in_arr, REAL_ft[:] out_arr, double fct) nogil except -1
