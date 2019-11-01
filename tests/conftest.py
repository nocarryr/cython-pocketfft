import pytest

FFT_LENGTHS = [256, 512, 1024, 4096, 8192]

@pytest.fixture(params=FFT_LENGTHS)
def fft_length(request):
    return request.param
