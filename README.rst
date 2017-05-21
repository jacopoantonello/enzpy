enzpy
=====

Implementation of the extended Nijboer-Zernike (**ENZ**) theory for Python.

This toolbox can be used to compute the point-spread function (**PSF**) using
the scalar **ENZ** theory, see [ENZ]_, [J2002]_, [B2008]_, and [H2010]_. It
also contains code to fit the phase and the generalised pupil function using
real- and complex-valued Zernike polynomials, see [A2015]_.


Main Features
-------------

* real- and complex-valued Zernike polynomials
* complex point-spread function computation
* multi-threaded computation of the :math:`V_n^m` terms (Eq.(2.48) in
  [B2008]_)
* routines to fit and evaluate the phase and the generalised pupil function
* load/save functions for each object
* numerous examples & documentation
* ENZPL algorithm example for phase retrieval


Requirements
------------

* `Python 3 <https://www.python.org>`__
* `pthreads
  <http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/pthread.h.html>`__
  (tested on Linux and Mac OS X Yosemite)
* `Numpy <http://www.numpy.org/>`__
* `h5py <http://www.h5py.org/>`__
* [optional] `matplotlib <http://matplotlib.org/>`__
* [optional] `PyQt5
  <http://www.riverbankcomputing.com/software/pyqt/download5>`__
* [optional] `CVXOPT
  <http://cvxopt.org>`__


Installation
------------

Linux
~~~~~
Make sure you have installed the packages in `Requirements`_.

.. code:: bash

    $ git clone https://github.com/jacopoantonello/enzpy.git
    $ cd enzpy
    $ sudo python setup.py install


Mac OS X
~~~~~~~~
The easiest way to use this toolbox is to install `Anaconda
<http://continuum.io/downloads>`__ for Python 3, which includes all the
necessary packages in `Requirements`_, except for PyQt5 and CVXOPT. Once you
have installed Anaconda, create an environment:

.. code:: bash

    $ conda create -n py3 python=3 anaconda
    $ source activate py3

and install `enzpy`:

.. code:: bash

    $ git clone https://github.com/jacopoantonello/enzpy.git
    $ cd enzpy
    $ python setup.py install

PyQt5 is necessary to run the examples with a graphical interface:
:code:`alpha_abs.py` and :code:`alpha_abs_qt.py`.


Examples
--------

After installing `enzpy`, you can run the examples located in `examples/`
(some screenshots are `here <http://www.antonello.org/code.php>`__):

* :code:`through_focus_intensity.py` is taken from [ENZ]_, and computes the
  intensity as a function of the radial coordinate and the defocus parameter.
* :code:`psf_plot.py` plots a diffraction-limited **PSF** at different
  defocus planes.
* :code:`phase_plot.py` plots the first 10 real-valued Zernike polynomials.
* :code:`fit_phase.py` estimates a vector of real-valued Zernike coefficients
  from a phase grid by taking inner products numerically.
* :code:`fit_gpf` estimates a vector of real-valued Zernike coefficients
  from a phase grid by taking inner products numerically. The coefficients can
  be used to approximate the generalised pupil function.
* :code:`beta_abs.py` and :code:`beta_abs_qt.py` plot the point-spread
  function that corresponds to a given complex-valued Zernike analysis of the
  generalised pupil function. The coefficients can be adjusted using the
  command line (:code:`beta_abs.py`) or a `Qt` widget
  (:code:`beta_abs_qt.py`).
* :code:`alpha_abs.py` and :code:`alpha_abs_qt.py` plot the point-spread
  function that corresponds to a given real-valued Zernike analysis of the
  phase aberration function. The coefficients can be adjusted using the command
  line (:code:`alpha_abs.py`) or a `Qt` widget (:code:`alpha_abs_qt.py`).
* :code:`enzpl/run` contains an example of the **ENZPL** algorithm, which
  uses **PhaseLift** (see [C2013]_) and the **ENZ** theory to correct a
  random aberration.

Alternatively, you can execute the consistency tests:

.. code:: bash

    $ cd tests
    $ nosetests -v -x --pdb *.py


References
----------

 .. [W1] http://www.antonello.org
 .. [ENZ] http://www.nijboerzernike.nl/
 .. [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach for the
    computation of optical point-spread functions," J. Opt. Soc. Am. A 19,
    849–857 (2002). `url <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
 .. [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
    Assessment of optical systems by means of point-spread functions,
    In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume 51,
    Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `url
    <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
 .. [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction
    Theory and its Applications (Ph.D. thesis, Delft University of
    Technology, The Netherlands, 2010). `url
    <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015). `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.
 .. [C2013] E. J. Candès, Y. C. Eldar, T. Strohmer, and V. Voroninski, "Phase
    retrieval via matrix completion," SIAM J. Imaging Sci. 6, 199–225 (2013).
    `url <http://dx.doi.org/10.1137/110848074>`__.
