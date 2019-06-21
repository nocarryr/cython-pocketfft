# cython: language_level=3
# distutils: include_dirs = cypocketfft/_pocketfft_lib
# distutils: sources = cypocketfft/_pocketfft_lib/pocketfft.c

cdef cfft_plan _make_cfft_plan(size_t length) nogil:
    return make_cfft_plan(length)

cdef void _destroy_cfft_plan(cfft_plan plan) nogil:
    destroy_cfft_plan(plan)

cdef int _cfft_backward(cfft_plan plan, double c[], double fct) nogil:
    return cfft_backward(plan, c, fct)

cdef int _cfft_forward(cfft_plan plan, double c[], double fct) nogil:
    return cfft_forward(plan, c, fct)

cdef size_t _cfft_length(cfft_plan plan) nogil:
    return cfft_length(plan)

cdef rfft_plan _make_rfft_plan(size_t length) nogil:
    return make_rfft_plan(length)

cdef void _destroy_rfft_plan(rfft_plan plan) nogil:
    destroy_rfft_plan(plan)

cdef int _rfft_backward(rfft_plan plan, double c[], double fct) nogil:
    return rfft_backward(plan, c, fct)

cdef int _rfft_forward(rfft_plan plan, double c[], double fct) nogil:
    return rfft_forward(plan, c, fct)

cdef size_t _rfft_length(rfft_plan plan) nogil:
    return rfft_length(plan)
