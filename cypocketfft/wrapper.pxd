# cython: language_level=3
# distutils: include_dirs = cypocketfft/_pocketfft_lib
# distutils: sources = cypocketfft/_pocketfft_lib/pocketfft.c

cdef extern from "pocketfft.h" nogil:
    cdef struct cfft_plan_i
    ctypedef cfft_plan_i * cfft_plan

    cfft_plan make_cfft_plan (size_t length)
    void destroy_cfft_plan (cfft_plan plan)
    int cfft_backward(cfft_plan plan, double c[], double fct)
    int cfft_forward(cfft_plan plan, double c[], double fct)
    size_t cfft_length(cfft_plan plan)

    cdef struct rfft_plan_i
    ctypedef rfft_plan_i * rfft_plan

    rfft_plan make_rfft_plan (size_t length)
    void destroy_rfft_plan (rfft_plan plan)
    int rfft_backward(rfft_plan plan, double c[], double fct)
    int rfft_forward(rfft_plan plan, double c[], double fct)
    size_t rfft_length(rfft_plan plan)

cdef cfft_plan _make_cfft_plan(size_t length) nogil
cdef void _destroy_cfft_plan(cfft_plan plan) nogil
cdef int _cfft_backward(cfft_plan plan, double c[], double fct) nogil
cdef int _cfft_forward(cfft_plan plan, double c[], double fct) nogil
cdef size_t _cfft_length(cfft_plan plan) nogil

cdef rfft_plan _make_rfft_plan(size_t length) nogil
cdef void _destroy_rfft_plan(rfft_plan plan) nogil
cdef int _rfft_backward(rfft_plan plan, double c[], double fct) nogil
cdef int _rfft_forward(rfft_plan plan, double c[], double fct) nogil
cdef size_t _rfft_length(rfft_plan plan) nogil

# fct: scale factor? ``1 / sqrt(N)`` if unnormilized

# typedef struct cmplx {
#   double r,i;
# } cmplx;
# typedef struct cfftp_fctdata
#   {
#   size_t fct;
#   cmplx *tw, *tws;
#   } cfftp_fctdata;

# typedef struct cfftp_plan_i
#   {
#   size_t length, nfct;
#   cmplx *mem;
#   cfftp_fctdata fct[NFCT];
#   } cfftp_plan_i;
# typedef struct cfftp_plan_i * cfftp_plan;

# typedef struct rfftp_fctdata
#   {
#   size_t fct;
#   double *tw, *tws;
#   } rfftp_fctdata;

# typedef struct rfftp_plan_i
#   {
#   size_t length, nfct;
#   double *mem;
#   rfftp_fctdata fct[NFCT];
#   } rfftp_plan_i;
# typedef struct rfftp_plan_i * rfftp_plan;

# static int cfftp_forward(cfftp_plan plan, double c[], double fct)
#   { return pass_all(plan,(cmplx *)c, fct, -1); }
# static int cfftp_backward(cfftp_plan plan, double c[], double fct)
#   { return pass_all(plan,(cmplx *)c, fct, 1); }
