import os
import sys
import json
from pathlib import Path
from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.errors import CCompilerError

RTFD_BUILD = 'READTHEDOCS' in os.environ.keys()

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
elif RTFD_BUILD:
    USE_CYTHON = True
else:
    USE_CYTHON = False

if USE_CYTHON:
    from Cython.Build import cythonize

CYTHON_TRACE = '--cython-trace' in sys.argv
if CYTHON_TRACE:
    sys.argv.remove('--cython-trace')
    if not USE_CYTHON:
        from Cython.Build import cythonize
    EXT_MACROS = [('CYTHON_TRACE_NOGIL', '1'), ('CYTHON_TRACE', '1')]
else:
    EXT_MACROS = []

PROJECT_PATH = Path(__file__).parent
WIN32 = sys.platform == 'win32'
IS_BUILD = len({'sdist', 'bdist_wheel', 'build_ext'} & set(sys.argv)) > 0
INCLUDE_PATH = ['src/cypocketfft/_pocketfft_lib']

try:
    import numpy
except ImportError:
    numpy = None
if numpy is not None:
    INCLUDE_PATH.append(numpy.get_include())

class CyBuildError(CCompilerError):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return '{}  (try building with "--use-cython")'.format(self.msg)


def get_cython_metadata(src_file):
    """Read the distutils metadata embedded in cythonized sources

    The JSON-formatted "Cython Metadata" contains all of the necessary compiler
    and linker options discovered by "cythonize()" to be passed as kwargs to
    recreate Extension objects without Cython installed

    ::
        /* BEGIN: Cython Metadata
            {
                "distutils": {
                    "depends": ["..."],
                    "extra_compile_args": ["..."],
                    "include_dirs": ["..."],
                    "language": "c/c++",
                    "name": "...",
                    "sources": ["..."]
                }
            }
        END: Cython Metadata */

    """
    if not isinstance(src_file, Path):
        src_file = Path(src_file)
    if src_file.suffix not in ['.c', '.cpp']:
        return None
    start_found = False
    end_found = False
    meta_lines = []
    i = -1
    with src_file.open('rt') as f:
        for line in f:
            i += 1
            if not start_found:
                if 'BEGIN: Cython Metadata' in line:
                    start_found = True
                elif i > 100:
                    return None
            else:
                if 'END: Cython Metadata' in line:
                    end_found = True
                    break
                meta_lines.append(line)
    if not end_found or not len(meta_lines):
        return None
    s = '\n'.join(meta_lines)
    return json.loads(s)

def build_extensions(pkg_dir, search_pattern='**/*.pyx'):
    """Create Extension objects from ".c" and ".cpp" sources built by Cython

    Searches for ".pyx" files in ``pkg_dir`` and their corresponding ".c" or ".cpp"
    files. ``get_cython_metadata()`` is then used to initialize the Extension

    """
    if not isinstance(pkg_dir, Path):
        pkg_dir = Path(pkg_dir)
    extensions = []
    names = set()
    for pyx_file in pkg_dir.glob(search_pattern):
        src_file = None
        for suffix in ['.c', '.cpp']:
            src_file = pyx_file.with_suffix(suffix)
            if src_file.exists():
                break
            else:
                src_file = None
        if src_file is None:
            raise CyBuildError('Could not find valid source for "{}"'.format(pyx_file))
        meta = get_cython_metadata(src_file)
        if meta is None:
            raise CyBuildError('Could not read metadata from source "{}"'.format(src_file))
        ext_name = meta['module_name']
        assert ext_name not in names
        names.add(ext_name)
        extensions.append(Extension(**meta['distutils']))
    return extensions

if USE_CYTHON:
    ext_modules = [
        Extension(
            'cypocketfft.fft',
            ['src/cypocketfft/fft.pyx'],
            include_dirs=INCLUDE_PATH,
            define_macros=EXT_MACROS,
        ),
        Extension(
            'cypocketfft.plancache',
            ['src/cypocketfft/plancache.pyx'],
            include_dirs=INCLUDE_PATH,
            define_macros=EXT_MACROS,
        ),
        Extension(
            'cypocketfft.wrapper',
            ['src/cypocketfft/wrapper.pyx', 'src/cypocketfft/_pocketfft_lib/pocketfft.c'],
            include_dirs=INCLUDE_PATH,
            extra_compile_args=['-std=c99'],
            extra_link_args=['-std=c99'],
            define_macros=EXT_MACROS,
        )
    ]
    ext_modules = cythonize(
        ext_modules,
        annotate=True,
        compiler_directives={
            'embedsignature':True,
            'linetrace':CYTHON_TRACE,
            'annotation_typing':False,
        },
    )
else:
    ext_modules = build_extensions(PROJECT_PATH / 'src' / 'cypocketfft')

setup(
    ext_modules=ext_modules,
    include_package_data=True,
)
