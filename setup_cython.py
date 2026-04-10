"""Build Cython extensions.

Usage:
    python setup_cython.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "swipealot.decoder._beam_search",
        sources=["src/swipealot/decoder/_beam_search.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
