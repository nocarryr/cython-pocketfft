
def get_include():
    import pkg_resources
    p = pkg_resources.resource_filename('cypocketfft', '_pocketfft_lib')
    return p
