:mod:`cypocketfft.fft`
======================

.. highlight:: cython

Python API
----------

.. automodule:: cypocketfft.fft
    :members:

C API
-----


Helpers
^^^^^^^


.. c:function:: size_t _rfft_length(double[:] in_arr)

    Calculate the rfft result length for the input array

    This evaluates to :math:`N / 2 + 1`

    :param in_arr: Input array or typed-memoryview (real-valued)
    :type in_arr: :c:type:`double` :c:type:`[:] <pyx_memoryview>`
    :return: The result length

.. c:function:: size_t _irfft_length(complex_t[:] in_arr)

    Calculate the irfft result length for the input array

    This evaluates to :math:`(N-1) * 2`

    :param in_arr: Input array (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :return: The result length


Real FFTs
^^^^^^^^^


.. c:function:: Py_ssize_t _rfft(double[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True)

    Perform the rfft

    :param in_arr: Input array or typed-memoryview (real-valued)
    :type in_arr: :c:type:`double` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of complex to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically ``1.0``)
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _rfft_with_plan(rfft_plan* plan, double[:] in_arr, complex_t[:] out_arr, double fct)

    Perform the rfft with an existing :c:type:`cypocketfft.wrapper.rfft_plan`

    :param plan: Pointer to a :c:type:`cypocketfft.wrapper.rfft_plan`
    :type plan: rfft_plan*
    :param in_arr: Input array or typed-memoryview (real-valued)
    :type in_arr: :c:type:`double` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of complex to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically ``1.0``)
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _irfft(complex_t[:] in_arr, double[:] out_arr, double fct, bint use_cache=True)

    Perform the inverse rfft

    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of double to store the result
    :type out_arr: :c:type:`double` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically :math:`1.0 / N`)
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _irfft_with_plan(rfft_plan* plan, complex_t[:] in_arr, double[:] out_arr, double fct)

    Perform the inverse rfft with an existing :c:type:`cypocketfft.wrapper.rfft_plan`

    :param plan: Pointer to a :c:type:`cypocketfft.wrapper.rfft_plan`
    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of double to store the result
    :type out_arr: :c:type:`double` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically :math:`1.0 / N`)


Complex FFTs
^^^^^^^^^^^^


.. c:function:: Py_ssize_t _cfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True)

    Perform the complex fft

    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of complex to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically ``1.0``)
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _icfft(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint use_cache=True)

    Perform the inverse complex fft

    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of :c:type:`double complex` to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform (typically :math:`1.0 / N`)
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _cfft_execute(complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=True, bint use_cache=True)

    Helper function called by :c:func:`_cfft` and c:func:`_icfft` to perform the
    forward or backward transform

    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of :c:type:`double complex` to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform
    :param bool is_forward: If ``True``, perform the forward fft. If ``False``, the inverse
    :param bool use_cache: If ``True``, use the built-in plancache

.. c:function:: Py_ssize_t _cfft_with_plan(cfft_plan* plan, complex_t[:] in_arr, complex_t[:] out_arr, double fct, bint is_forward=True)

    Perform the forward or inverse complex fft with an existing :c:type:`cypocketfft.wrapper.cfft_plan`

    :param plan: Pointer to a :c:type:`cypocketfft.wrapper.cfft_plan`
    :param in_arr: Input array or typed-memoryview (complex-valued)
    :type in_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param out_arr: Pre-allocated buffer of :c:type:`double complex` to store the result
    :type out_arr: :c:type:`complex_t` :c:type:`[:] <pyx_memoryview>`
    :param double fct: Scaling factor to apply to the un-normalized transform
    :param bool is_forward: If ``True``, perform the forward fft. If ``False``, the inverse
