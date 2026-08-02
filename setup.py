#!/usr/bin/env python
# encoding: utf-8

from distutils.extension import Extension

import numpy as np
from Cython.Build import build_ext, cythonize
from setuptools import setup


def extension(name, source):
    """Create a Cython extension using NumPy's supported public C API."""
    return Extension(
        name,
        [source],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )


extensions = [
    extension('madmom.ml.hmm', 'madmom/ml/hmm.pyx'),
    extension('madmom.ml.nn.layers', 'madmom/ml/nn/layers.py'),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': 3},
    ),
    cmdclass={'build_ext': build_ext},
)
