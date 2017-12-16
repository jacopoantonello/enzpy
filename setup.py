#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# enzpy - Extended Nijboer-Zernike implementation for Python
# Copyright 2016-2018 J. Antonello <jacopo@antonello.org>
#
# This file is part of enzpy.
#
# enzpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# enzpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with enzpy.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from os import path


# see http://stackoverflow.com/questions/19919905/
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='enzpy',
    version='1.0.0',
    description='Extended Nijboer-Zernike implementation for Python',
    long_description=long_description,
    url='https://github.com/jacopoantonello/enzpy',
    author='Jacopo Antonello',
    author_email='jack@antonello.org',
    license='GPLv3+',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics', (
            'License :: OSI Approved :: GNU General Public License v3 ' +
            'or later (GPLv3+)'),
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX'
    ],
    packages=find_packages(exclude=['tests*', 'examples*']),
    setup_requires=['numpy'],
    install_requires=['numpy', 'h5py'],
    extras_require={
        'user interface': ['pyqt5'],
        'plot': ['matplotlib'],
        'phase retrieval': ['cvxopt']
        },
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(
        'enzpy._enz',
        define_macros=[('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0')],
        libraries=['m', 'pthread'],
        extra_compile_args=['-Wall'],
        sources=['enzpy/_enz/_enz.c', 'enzpy/_enz/vnmpocnp.c'])]
)
