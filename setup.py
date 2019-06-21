import sys
from pathlib import Path
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Build.Dependencies import default_create_extension

PROJECT_PATH = Path(__file__).parent
LIB_PATH = Path('.') / 'cypocketfft/_pocketfft_lib'
WIN32 = sys.platform == 'win32'
IS_BUILD = len({'sdist', 'bdist_wheel', 'build_ext'} & set(sys.argv)) > 0

ext_modules = [
    # Extension('cypocketfft._pocketfft_lib.pocketfft', [str(LIB_PATH / 'pocketfft.c')], include_dirs=[str(LIB_PATH)]),
]

ext_modules += cythonize(
    ['cypocketfft/**/*.pyx'],
    include_path=[LIB_PATH],
    annotate=True,
    # language='c++',
    # gdb_debug=True,
    compiler_directives={
        'linetrace':True,
        'embedsignature':True,
        'binding':True,
    },
    # create_extension=my_create_extension,
)

setup(
    ext_modules=ext_modules,
    include_package_data=True,
)
