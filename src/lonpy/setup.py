import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("sampling_run.pyx", annotate=True),
    # Include numpy headers so Cython can find them
    include_dirs=[np.get_include()]
)