import sys
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

import cypocketfft

if sys.platform == 'darwin':
    COMPILE_ARGS = []
else:
    COMPILE_ARGS = ['-fopenmp']
INCLUDE_PATH = [cypocketfft.get_include()]

ext_modules = [
    Extension(
        '_test_plancache', ['_test_plancache.pyx'],
        include_dirs=INCLUDE_PATH,
    ),
    Extension(
        '_testfft', ['_testfft.pyx'],
        include_dirs=INCLUDE_PATH,
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=COMPILE_ARGS,
    ),
]

setup(
    name='cython-pocketfft-test',
    packages=['.'],
    scripts=[],
    ext_modules=cythonize(ext_modules),
)
