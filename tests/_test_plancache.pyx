# cython: language_level=3
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import time
import threading

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uintptr_t

from cypocketfft.wrapper cimport *
from cypocketfft cimport fft
from cypocketfft.fft cimport COMPLEX_ft
from cypocketfft.plancache cimport PlanCache

import numpy as np
cimport numpy as np

def LOG(s):
    print(s)
    time.sleep(.1)

def test():
    cdef PlanCache _plan_cache = PlanCache()

    cdef size_t length
    cdef rfft_plan plan
    cdef rfft_plan plan2

    length = 8
    plan = _plan_cache.get_rplan(length, False)
    assert _plan_cache.rfft_plans.count(length) == 0

    LOG('adding plan: length={}'.format(length))
    plan = _plan_cache.get_rplan(length)
    LOG('plan added')
    assert _plan_cache.rfft_plans.count(length) == 1

    LOG('getting existing plan: length={}'.format(length))
    plan2 = _plan_cache.get_rplan(length)
    LOG('plan retrieved')

    assert plan == plan2

    LOG('__dealloc__')

    plan = NULL
    plan2 = NULL
    del _plan_cache

    LOG('done')

cdef class State:
    cdef readonly bint ready, complete
    cdef readonly size_t nthreads, num_complete
    def __cinit__(self, size_t nthreads):
        self.nthreads = nthreads
        self.num_complete = 0
        self.ready = False
        self.complete = False
    cdef void on_thread_complete(self) nogil:
        if self.num_complete == 0:
            self.complete = True
            return
        self.num_complete -= 1
        if self.num_complete == 0:
            self.complete = True
    cdef void set_ready(self) nogil:
        self.ready = True

cdef class PlanCreate:
    cdef readonly State state
    cdef PlanCache plan_cache
    cdef size_t length_start, nplans, nplans_built
    cdef rfft_plan *plans
    cdef readonly object thread
    def __cinit__(self, State state, PlanCache plan_cache, size_t length_start, size_t nplans):
        self.state = state
        self.plan_cache = plan_cache
        self.length_start = length_start
        self.nplans = nplans
        self.nplans_built = 0
        self.plans = <rfft_plan *>PyMem_Malloc(sizeof(rfft_plan) * nplans)
        if not self.plans:
            raise MemoryError()
    def __init__(self, *args):
        self.thread = PlanCreateThread(self)
        self.thread.start()
        self.thread.running.wait()
    def __dealloc__(self):
        cdef rfft_plan *plans
        if self.plans != NULL:
            plans = self.plans
            self.plans = NULL
            PyMem_Free(plans)
    @property
    def ready(self):
        return self.state.ready

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _build_plans(self) nogil except *:
        cdef Py_ssize_t i, nplans = self.nplans, length_start = self.length_start
        cdef size_t length
        # cdef PlanCache plan_cache = self.plan_cache
        cdef rfft_plan *plans = self.plans
        cdef rfft_plan plan

        for i in range(nplans):
            length = length_start + i
            plan = self.plan_cache.get_rplan(length)
            if plan == NULL:
                with gil:
                    raise Exception('NULL plan')
            plans[i] = plan
            self.nplans_built += 1

    cpdef build_plans(self):
        with nogil:
            self._build_plans()
            self.state.on_thread_complete()
    cdef check_plan(self, rfft_plan plan, size_t length):
        cdef double[:] x = np.random.uniform(-1., 1., length)
        cdef size_t ff_length = fft._rfft_length(x)
        cdef double complex[:] ff = np.empty(ff_length, dtype=np.complex128)
        cdef double complex[:] ff2 = np.empty(ff_length, dtype=np.complex128)

        fft._rfft_with_plan(&plan, x, ff, 1.0)
        fft._rfft(x, ff2, 1.0)

        assert not np.all(np.equal(ff, 0))
        assert np.array_equal(ff, ff2)

    def check(self):
        assert self.nplans_built == self.nplans
        cdef size_t length
        cdef rfft_plan plan
        cdef rfft_plan cached_plan
        cdef uintptr_t plan_ptr, cached_ptr

        for i in range(self.nplans):
            length = self.length_start + i
            plan = self.plans[i]
            cached_plan = self.plan_cache.get_rplan(length)
            plan_ptr = <uintptr_t>plan
            cached_ptr = <uintptr_t>cached_plan
            assert plan_ptr == cached_ptr
            assert plan == cached_plan
            self.check_plan(plan, length)

class PlanCreateThread(threading.Thread):
    def __init__(self, plan_create):
        super().__init__()
        self.plan_create = plan_create
        self.running = threading.Event()
        self.stopped = threading.Event()
        self.exc = None
        self.daemon = True
    def run(self):
        self.running.set()
        while self.running.is_set():
            if not self.plan_create.ready:
                time.sleep(.1)
                continue
            try:
                self.plan_create.build_plans()
            except Exception as exc:
                self.exc = exc
            break
        self.running.clear()
        self.stopped.set()
    def stop(self):
        self.running.clear()
        self.stopped.wait()

def test_concurrency(size_t nthreads, size_t nplans):
    cdef PlanCache plan_cache = PlanCache()
    cdef State state = State(nthreads)
    cdef PlanCreate creator
    cdef size_t length_start = 128

    creators = []
    for i in range(nthreads):
        creator = PlanCreate(state, plan_cache, length_start, nplans)
        creators.append(creator)

    LOG('creators built')

    state.set_ready()
    start_ts = time.time()
    end_ts = start_ts + 60
    timed_out = False
    while not state.complete:
        time.sleep(.1)
        ts = time.time()
        if ts >= end_ts:
            timed_out = True
            break
    if timed_out:
        del creators
        raise Exception('Timeout')

    LOG('checking results')
    for creator in creators:
        exc = creator.thread.exc
        if exc is not None:
            raise exc
        creator.thread.stopped.wait()
        creator.thread.join()
        creator.check()
    LOG('dealloc creators')
    del creators
    LOG('dealloc plan_cache')
    del plan_cache
    LOG('complete')
