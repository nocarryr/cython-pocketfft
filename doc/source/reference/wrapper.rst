:mod:`cypocketfft.wrapper`
==========================

.. highlight:: cython

C-API
-----

.. c:type:: size_t

.. c:type:: cfft_plan

.. c:type:: rfft_plan

.. c:function:: cfft_plan _make_cfft_plan(size_t length)

    Prepare a plan for complex fft functions

    :param length: The input length for the plan
    :type length size_t:
    :return: A C struct with plan data
    :rtype: cfft_plan


.. c:function:: void _destroy_cfft_plan(cfft_plan plan)

.. c:function:: int _cfft_backward(cfft_plan plan, double c[], double fct)

.. c:function:: _cfft_length(cfft_plan plan)

.. c:function:: rfft_plan _make_rfft_plan(size_t length)

.. c:function:: void _destroy_rfft_plan(rfft_plan plan)
