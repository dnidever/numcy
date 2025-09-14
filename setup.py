from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "numcy.fast_array_nd",
        sources=["numcy/fastarray.pyx"],
        include_dirs=[np.get_include()],
        libraries=["blas"],  # or ["openblas"] depending on your system
        library_dirs=[],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "numcy.utils",
        sources=["numcy/utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="numcy",
    version="0.1.0",
    description="Fast Cython-based NumPy-like arrays for high-performance computing",
    author="Your Name",
    packages=["numcy"],
    ext_modules=cythonize(extensions, annotate=True, 
                          compiler_directives={'language_level': 3}),
    zip_safe=False,
)

