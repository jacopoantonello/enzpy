#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


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

setup(name='enzpy',
      version='1.0.3',
      description='Extended Nijboer-Zernike implementation for Python',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jacopoantonello/enzpy',
      author='Jacopo Antonello',
      author_email='jack@antonello.org',
      license='GPLv3+',
      classifiers=[
          'Development Status :: 4 - Beta', 'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Physics',
          ('License :: OSI Approved :: GNU General Public License v3 ' +
           'or later (GPLv3+)'), 'Programming Language :: Python :: 3',
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
      ext_modules=[
          Extension('enzpy._enz',
                    define_macros=[('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                    libraries=['m', 'pthread'],
                    extra_compile_args=['-Wall'],
                    sources=['enzpy/_enz/_enz.c', 'enzpy/_enz/vnmpocnp.c'])
      ])
