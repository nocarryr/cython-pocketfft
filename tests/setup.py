from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

import cypocketfft

INCLUDE_PATH = [cypocketfft.get_include()]

ext_modules = [
    Extension(
        '_test_plancache', ['_test_plancache.pyx'],
        include_dirs=INCLUDE_PATH,
    ),
    Extension(
        '_testfft', ['_testfft.pyx'],
        include_dirs=INCLUDE_PATH,
    ),
]

setup(
    name='cython-pocketfft-test',
    packages=['.'],
    scripts=[],
    ext_modules=cythonize(ext_modules),
)
