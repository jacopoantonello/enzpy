# enzpy

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FJOSAA.32.001160-blue)](http://dx.doi.org/10.1364/JOSAA.32.001160)

Implementation of the extended Nijboer-Zernike (**ENZ**) theory for Python.
This toolbox can be used to compute the point-spread function (**PSF**) using
the scalar **ENZ** theory, see [[1]](#1), [[2]](#2), and [[3]](#3). It contains
code to fit the phase and the generalised pupil function using real- and
complex-valued Zernike polynomials, see [[4]](#4).


## Main Features

* real- and complex-valued Zernike polynomials
* complex point-spread function computation
* multi-threaded computation of the :math:`V_n^m` terms (Eq.(2.48) in
  [[3]](#3))
* routines to fit and evaluate the phase and the generalised pupil function
* load/save functions for each object
* numerous examples & documentation
* ENZPL algorithm example for phase retrieval


## Requirements

* [Python 3](https://www.python.org)
* [pthreads](http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/pthread.h.html)
  (tested on Linux and Mac OS X Yosemite)
* [Numpy](http://www.numpy.org)
* [h5py](http://www.h5py.org)
* [optional] [matplotlib](http://matplotlib.org)
* [optional] [PyQt5](http://www.riverbankcomputing.com/software/pyqt/download5)
* [optional] [CVXOPT](http://cvxopt.org)


## Installation

### Linux

Make sure you have installed the packages in requirements.

```bash
git clone https://github.com/jacopoantonello/enzpy.git
cd enzpy
sudo python setup.py install
```


### Mac OS X

The easiest way to use this toolbox is to install
[Anaconda](http://continuum.io/downloads) for Python 3, which includes all the
necessary packages in requirements, except for PyQt5 and CVXOPT. Once you
have installed Anaconda, create an environment:

```bash
conda create -n py3 python=3 anaconda
source activate py3
```

and install `enzpy`:

```bash
git clone https://github.com/jacopoantonello/enzpy.git
cd enzpy
python setup.py install
```

PyQt5 is necessary to run the examples with a graphical interface:
`alpha_abs.py` and `alpha_abs_qt.py`.


### Windows

This toolbox **does not support** Windows.


## Examples

After installing `enzpy`, you can run the examples located in `examples/`:

* `through_focus_intensity.py` is taken from [[1]](#1), and computes the
  intensity as a function of the radial coordinate and the defocus parameter.
* `psf_plot.py` plots a diffraction-limited **PSF** at different
  defocus planes.
* `phase_plot.py` plots the first 10 real-valued Zernike polynomials.
* `fit_phase.py` estimates a vector of real-valued Zernike coefficients
  from a phase grid by taking inner products numerically.
* `fit_gpf` estimates a vector of real-valued Zernike coefficients
  from a phase grid by taking inner products numerically. The coefficients can
  be used to approximate the generalised pupil function.
* `beta_abs.py` and `beta_abs_qt.py` plot the point-spread
  function that corresponds to a given complex-valued Zernike analysis of the
  generalised pupil function. The coefficients can be adjusted using the
  command line (`beta_abs.py`) or a `Qt` widget
  (`beta_abs_qt.py`).
* `alpha_abs.py` and `alpha_abs_qt.py` plot the point-spread
  function that corresponds to a given real-valued Zernike analysis of the
  phase aberration function. The coefficients can be adjusted using the command
  line (`alpha_abs.py`) or a `Qt` widget (`alpha_abs_qt.py`).
* `enzpl/run` contains an example of the **ENZPL** algorithm, which
  uses **PhaseLift** (see [[5]](#5)) and the **ENZ** theory to correct a
  random aberration.

Alternatively, you can execute the consistency tests:

```bash
cd tests
nosetests -v -x --pdb *.py
```


## References

<a id="1">[1]</a> [nijboerzernike.nl](http://www.nijboerzernike.nl)

<a id="2">[2]</a> A. J. E. M. Janssen, "Extended Nijboer–Zernike approach for the computation of optical point-spread functions," J. Opt. Soc. Am. A [19](http://dx.doi.org/10.1364/JOSAA.19.000849), 849–857 (2002)

<a id="3">[3]</a> J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6 Assessment of optical systems by means of point-spread functions, In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume [51](http://dx.doi.org/10.1016/S0079-6638(07)51006-1), Pages 349-468, ISSN 0079-6638, ISBN 9780444532114

<a id="4">[4]</a> J. Antonello and M. Verhaegen, "Modal-based phase retrieval for adaptive optics," J. Opt. Soc. Am. A [32](http://dx.doi.org/10.1364/JOSAA.32.001160), 1160-1170 (2015)

<a id="5">[5]</a> E. J. Candès, Y. C. Eldar, T. Strohmer, and V. Voroninski, "Phase retrieval via matrix completion," SIAM J. Imaging Sci. [6](http://dx.doi.org/10.1137/110848074), 199–225 (2013)
