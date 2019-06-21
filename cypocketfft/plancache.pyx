# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11


from cypocketfft cimport wrapper

cdef void rfft_plan_map_destroy(RFFT_PLAN_MAP_t plans) except *:
    cdef rfft_plan plan
    for pair in plans:
        if pair.second != NULL:
            plan = pair.second
            pair.second = NULL
            wrapper._destroy_rfft_plan(plan)

cdef void cfft_plan_map_destroy(CFFT_PLAN_MAP_t plans) except *:
    cdef cfft_plan plan
    for pair in plans:
        if pair.second != NULL:
            plan = pair.second
            pair.second = NULL
            wrapper._destroy_cfft_plan(plan)

cdef class PlanCache:
    def __dealloc__(self):
        rfft_plan_map_destroy(self.rfft_plans)
        cfft_plan_map_destroy(self.cfft_plans)
    cdef rfft_plan get_rplan(self, size_t length, bint create=True) nogil except *:
        cdef rfft_plan plan
        if self.rfft_plans.count(length) == 0:
            if not create:
                return NULL
            plan = self._build_rplan(length)
        else:
            plan = self.rfft_plans[length]
            self.rfft_plans.insert(RFFT_PLAN_ITEM_t(length, plan))
        return plan
    cdef void del_rplan(self, size_t length) nogil except *:
        if self.rfft_plans.count(length) == 0:
            return
        cdef rfft_plan plan = self.rfft_plans[length]
        self.rfft_plans.erase(length)
        wrapper._destroy_rfft_plan(plan)
    cdef rfft_plan _build_rplan(self, size_t length) nogil except *:
        cdef rfft_plan plan = wrapper._make_rfft_plan(length)
        if not plan:
            with gil:
                raise MemoryError()
        return plan
    cdef cfft_plan get_cplan(self, size_t length, bint create=True) nogil except *:
        cdef cfft_plan plan
        if self.cfft_plans.count(length) == 0:
            plan = self._build_cplan(length)
            if not create:
                return NULL
        else:
            plan = self.cfft_plans[length]
            self.cfft_plans.insert(CFFT_PLAN_ITEM_t(length, plan))
        return plan
    cdef cfft_plan _build_cplan(self, size_t length) nogil except *:
        cdef cfft_plan plan = wrapper._make_cfft_plan(length)
        if not plan:
            with gil:
                raise MemoryError()
        return plan
    cdef void del_cplan(self, size_t length) nogil except *:
        if self.cfft_plans.count(length) == 0:
            return
        cdef cfft_plan plan = self.cfft_plans[length]
        self.cfft_plans.erase(length)
        wrapper._destroy_cfft_plan(plan)

cdef PlanCache plan_cache = PlanCache()
