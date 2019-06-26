# cython: language_level=3

from libcpp.unordered_map cimport unordered_map as cpp_map
from libcpp.utility cimport pair as cpp_pair
from cypocketfft.wrapper cimport cfft_plan, rfft_plan

ctypedef cpp_map[size_t, rfft_plan] RFFT_PLAN_MAP_t
ctypedef cpp_pair[size_t, rfft_plan] RFFT_PLAN_ITEM_t
ctypedef cpp_map[size_t, cfft_plan] CFFT_PLAN_MAP_t
ctypedef cpp_pair[size_t, cfft_plan] CFFT_PLAN_ITEM_t

cdef extern from "<mutex>" namespace "std" nogil:
    cdef cppclass mutex:
        mutex() except +
        mutex(const mutex&) except +
        mutex& operator=(const mutex&)# = delete

        void lock()
        bint try_lock()
        void unlock()


cdef class PlanCache:
    cdef RFFT_PLAN_MAP_t rfft_plans
    cdef CFFT_PLAN_MAP_t cfft_plans
    cdef mutex _lock

    cdef rfft_plan get_rplan(self, size_t length, bint create=*) nogil except *
    cdef void del_rplan(self, size_t length) nogil except *
    cdef cfft_plan get_cplan(self, size_t length, bint create=*) nogil except *
    cdef void del_cplan(self, size_t length) nogil except *

    cdef rfft_plan _build_rplan(self, size_t length) nogil except *
    cdef cfft_plan _build_cplan(self, size_t length) nogil except *

cdef PlanCache plan_cache
