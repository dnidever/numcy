from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="numcy.fastarray",
        sources=["numcy/fastarray.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
]

setup(
    name="numcy",
    version="0.1.0",
    description="Fast array operations in Cython",
    packages=["numcy"],
    ext_modules=cythonize(ext_modules, annotate=True, language_level=3),
    python_requires=">=3.8",
)
