#! /usr/bin/env python
# coding:utf-8

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="plsi library",
    ext_modules=cythonize("*.pyx")
)
