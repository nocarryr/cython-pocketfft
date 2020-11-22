Types
=====

.. currentmodule:: cypocketfft

.. c:type:: double

    64-bit float

.. c:type:: complex_t

    128-bit :class:`complex` type. Alias for :c:type:`double complex`

.. c:type:: pyx_memoryview

    Array-like object or buffer. Can be :class:`numpy.ndarray`, a `Cython Array`_,
    a C array or anything supported by `Cython Typed Memoryviews`_

.. _Cython Typed Memoryviews: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
.. _Cython Array: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#view-cython-arrays
