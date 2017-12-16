#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Complex point-spread function using the Extended Nijboer-Zernike theory.

"""

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

import numpy as np
import h5py
import math
import cmath

from enzpy._enz import vnmpocnp
from enzpy.czernike import CZern


__docformat__ = 'restructuredtext'
HDF5_options = {
    'chunks': True,
    'shuffle': True,
    'fletcher32': True,
    'compression': 'gzip',
    'compression_opts': 9}


def get_field_unit(wavelength, aperture_radius, exit_pupil_sphere_radius):
    """Compute the diffraction unit, i.e., `wavelength`/`NA`."""
    return wavelength/(aperture_radius/exit_pupil_sphere_radius)


def get_focus_param(defocus):
    """Convert the Zernike `defocus` to the corresponding defocus parameter."""
    return 2.0*math.sqrt(3.0)*defocus


def get_defocus(defocus_param):
    """Convert `defocus_param` to the corresponding Zernike defocus."""
    return defocus_param/(2.0*math.sqrt(3.0))


class CPsf:
    r"""Complex point-spread function object.

    A finite set of complex-valued Zernike polynomials are used to approximate
    the generalised pupil function.  The polynomials are ordered and normalised
    as outlined in Appendix A of [A2015]_. The complex point-spread function is
    computed using the extended Nijboer-Zernike formulas found in [B2008]_. The
    coordinates in the image plane :math:`(r, \phi)` are normalised by the
    diffraction unit, see [B2008]_, [J2002]_, and [H2010]_.

    References
    ----------
    ..  [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach for
        the computation of optical point-spread functions," J. Opt. Soc.  Am.
        A 19, 849–857 (2002). `doi
        <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
    ..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
        Assessment of optical systems by means of point-spread functions, In:
        E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume 51,
        Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `doi
        <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
    ..  [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction Theory
        and its Applications (Ph.D. thesis, Delft University of Technology,
        The Netherlands, 2010). `doi
        <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.
    ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
        retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
        (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

    """

    def __init__(self, n_beta):
        r"""New `CPsf` with Zernike polynomials up to radial order `n_beta`.

        The number of polynomials is `self.czern.nk`, i.e.,
        :math:`N_\beta = (n_\beta + 1)(n_\beta + 2)/2`.

        """
        self.czern = CZern(n_beta)

    def P(self, beta, rho, theta):
        r"""Evaluate the generalised pupil function at a point.

        The generalised pupil function is

        .. math::

            P(\rho, \theta) = \sum_{n, m} \beta_n^m
                \sqrt{n + 1}R_n^{|m|}(\rho) exp(i m \theta).

        The summation extends over :math:`N_beta` = `self.czern.nk` addends.
        The support of :math:`P(\rho, \theta)` is the unit disk. See Eq. (2) in
        [A2015]_.

        Parameters
        ----------
        - `beta`: list of complex-valued Zernike coefficients :math:`\beta_n^m`
        - `rho`: float for the radial coordinate :math:`\rho`
        - `theta`: float for the azimuthal coordinate :math:`\theta`

        Returns
        -------
        - `P`: complex value of :math:`P(\rho, \theta)`

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

        """
        return self.czern.eval_a(beta, rho, theta)

    def _numpify(self, a, dtype):
        if isinstance(a, np.ndarray):
            return a.ravel().astype(dtype, order='F', copy=False)
        elif isinstance(a, list):
            return np.array(a, order='F', dtype=dtype)
        else:
            return np.array([a], order='F', dtype=dtype)

    def V(self, n, m, r, f, L_max=35, ncpus=-1):
        r"""Evaluate :math:`V_n^m(r, f)`.

        Compute

        .. math::

            V_n^m(r, f) = \epsilon_m \exp(if) \sum_{l=1}^{L_max}(-2if)^{l - 1}
                \sum_{j=0}^{(n - |m|)/2} v_{l,j}
                (1/l(2\pi r)^l)J_{|m| + l + 2j}(2\pi r).

        See Eq. (2.48) in [B2008]_.

        Parameters
        ----------
        -   `n`: `numpy` vector of integers for the radial orders `n`
        -   `m`: `numpy` vector of integers for the azimuthal frequencies `m`
        -   `r`: `numpy` vector of doubles for the radial coordinate :math:`r`,
            which is normalised to the diffraction unit `wavelength/NA`
        -   `f`: `numpy` vector of complex numbers for the defocus parameter,
            see [J2002]_, [B2008]_, and [H2010]_
        -   `L_max`: optional `int` for the truncation order of the series,
            see [J2002]_, [B2008]_, and [H2010]_. `L_max <= 0` uses the
            default value of `35`.
        -   `ncpus` : optional `int` for the number of threads. `-1` chooses
            all available cpus

        Returns
        -------
        -   `vnm`: `numpy` array of shape `(r.size, f.size, n.size)` for
            :math:`V_n^m(r, f)`

        References
        ----------
        ..  [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach
            for the computation of optical point-spread functions," J. Opt.
            Soc.  Am.  A 19, 849–857 (2002). `doi
            <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
        ..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
            Assessment of optical systems by means of point-spread functions,
            In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume
            51, Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `doi
            <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
        ..  [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction
            Theory and its Applications (Ph.D. thesis, Delft University of
            Technology, The Netherlands, 2010). `doi
            <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.

        """
        n = self._numpify(n, np.int)
        m = self._numpify(m, np.int)
        r = self._numpify(r, np.float)
        f = self._numpify(f, np.complex)

        Vnm = vnmpocnp(r, f, n, m, L_max=L_max, ncpus=ncpus)

        if Vnm.size == 1:
            return complex(Vnm[0, 0, 0])
        else:
            return Vnm

    def U(self, beta, r, phi, f):
        r"""Evaluate the complex point spread function at a point.

        The complex point-spread function is

        .. math::

            U(r, \phi, f) = 2 \sum_{n, m} \beta_{n}^{m}
                \sqrt{n + 1} i^{m} V_n^m(r, f) \exp(im\phi).

        See Eq. (4) in [A2015]_.

        Parameters
        ----------
        -   `beta`: `list` of the complex Zernike coefficients
            :math:`\beta_n^m`
        -   `r`: float for the radial coordinate :math:`r`, normalised to
            the diffraction unit `wavelength/NA`
        -   `phi`: float for the azimuthal coordinate :math:`\phi`
        -   `f`: complex-valued defocus parameter, see [J2002]_, [B2008]_, and
            [H2010]_

        Returns
        -------
        -   `U`: complex value of :math:`U(r, \phi, f)`

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.
        ..  [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach
            for the computation of optical point-spread functions," J. Opt.
            Soc.  Am.  A 19, 849–857 (2002). `doi
            <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
        ..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
            Assessment of optical systems by means of point-spread functions,
            In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume
            51, Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `doi
            <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
        ..  [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction
            Theory and its Applications (Ph.D. thesis, Delft University of
            Technology, The Netherlands, 2010). `doi
            <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.

        """
        U = complex(0.0)
        r = self._numpify(r, np.float)
        f = self._numpify(f, np.complex)
        assert(r.size == 1)
        assert(f.size == 1)
        n, m = np.array([0]), np.array([0])
        for i in range(self.czern.nk):
            n[0], m[0] = self.czern.ntab[i], self.czern.mtab[i]
            cf = self.czern.coefnorm[i]
            U += complex(
                beta[i]*cf*((1j)**m[0]) *
                vnmpocnp(r, f, n, m)[0][0][0]*cmath.exp(1j*m[0]*phi))
        return 2.0*U

    def _trim_r(self, r_sp, min_r):
        if np.abs(r_sp).min() < min_r:
            r_sp[np.abs(r_sp) <= min_r] = min_r
            assert(np.abs(r_sp).min() >= min_r)
        return r_sp

    def make_cart_grid(self, x_sp=None, y_sp=None, f_sp=None, min_r=1e-9):
        r"""Make a cartesian grid for the complex point-spread function.

        The complex point-spread function is

        .. math::

            U(r, \phi, f) = 2 \sum_{n, m} \beta_{n}^{m}
                \sqrt{n + 1} i^{m} V_n^m(r, f) \exp(im\phi).

        See Eq. (4) in [A2015]_.

        Parameters
        ----------
        -   `x_sp`: `numpy` array of the `x` coordinates in the image plane
        -   `y_sp`: `numpy` array of the `y` coordinates in the image plane
        -   `f_sp`: `numpy` array of complex numbers for the defocus parameter,
            see [J2002]_, [B2008]_, [H2010]_
        -   `min_r`: float, optional threshold to avoid division by zero in
            the image plane

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.
        ..  [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach
            for the computation of optical point-spread functions," J. Opt.
            Soc.  Am.  A 19, 849–857 (2002). `doi
            <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
        ..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
            Assessment of optical systems by means of point-spread functions,
            In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume
            51, Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `doi
            <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
        ..  [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction
            Theory and its Applications (Ph.D. thesis, Delft University of
            Technology, The Netherlands, 2010). `doi
            <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.

        """
        x_sp = self._numpify(x_sp, np.float).ravel(order='F')
        y_sp = self._numpify(y_sp, np.float).ravel(order='F')
        f_sp = self._numpify(f_sp, np.complex).ravel(order='F')

        xx, yy = np.meshgrid(x_sp, y_sp)
        r_sp = np.sqrt(np.square(xx) + np.square(yy)).ravel(order='F')

        # remove nans due to r = 0.0
        r_sp = self._trim_r(r_sp, min_r)

        ph_sp = np.arctan2(yy, xx).ravel(order='F')
        Ugrid = np.zeros(
            (x_sp.size, y_sp.size, f_sp.size, self.czern.nk),
            order='F', dtype=np.complex)
        Vnm = vnmpocnp(r_sp, f_sp, self.czern.ntab, self.czern.mtab)
        assert(np.all(np.isfinite(Vnm)))
        Cnm = np.zeros(
            (ph_sp.size, self.czern.nk), order='F', dtype=np.complex)
        for k in range(self.czern.nk):
            m = self.czern.mtab[k]
            Cnm[:, k] = 2.0*self.czern.coefnorm[k]*((1j)**m)*np.exp(1j*m*ph_sp)
        for k in range(self.czern.nk):
            for f in range(f_sp.size):
                Ugrid[:, :, f, k] = Vnm[:, f, k].reshape(
                        (x_sp.size, y_sp.size), order='F')*Cnm[:, k].reshape(
                                (x_sp.size, y_sp.size), order='F')
        assert(np.all(np.isfinite(Ugrid)))
        self.Ugrid = Ugrid
        self.Vnm = Vnm
        self.Cnm = Cnm

    def make_pol_grid(self, r_sp=None, ph_sp=None, f_sp=None, min_r=1e-9):
        r"""Make a polar grid for the complex point-spread function.

        The complex point-spread function is

        .. math::

            U(r, \phi, f) = 2 \sum_{n, m} \beta_{n}^{m}
                \sqrt{n + 1} i^{m} V_n^m(r, f) \exp(im\phi).

        See Eq. (4) in [A2015]_.

        Parameters
        ----------
        -   `r_sp`: `numpy` array of the radial coordinate :math:`r`
        -   `ph_sp`: `numpy` array of the azimuthal coordinate :math:`\phi`
        -   `f_sp`: `numpy` array of complex numbers for the defocus parameter,
            see [J2002]_, [B2008]_, [H2010]_
        -   `min_r`: float, optional threshold to avoid division by zero in
            the image plane

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.
        ..  [J2002] A. J. E. M. Janssen, "Extended Nijboer–Zernike approach
            for the computation of optical point-spread functions," J. Opt.
            Soc.  Am.  A 19, 849–857 (2002). `doi
            <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.
        ..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
            Assessment of optical systems by means of point-spread functions,
            In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume
            51, Pages 349-468, ISSN 0079-6638, ISBN 9780444532114. `doi
            <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.
        ..  [H2010] S. van Haver, The Extended Nijboer-Zernike Diffraction
            Theory and its Applications (Ph.D. thesis, Delft University of
            Technology, The Netherlands, 2010). `doi
            <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-1bc348155061>`__.

        """
        r_sp = self._numpify(r_sp, np.float).ravel(order='F')
        ph_sp = self._numpify(ph_sp, np.float).ravel(order='F')
        f_sp = self._numpify(f_sp, np.complex).ravel(order='F')

        # remove nans due to r = 0.0
        r_sp = self._trim_r(r_sp, min_r)

        Ugrid = np.zeros(
            (r_sp.size, ph_sp.size, f_sp.size, self.czern.nk), order='F',
            dtype=np.complex)
        Vnm = vnmpocnp(r_sp, f_sp, self.czern.ntab, self.czern.mtab)
        Cnm = np.zeros(
            (ph_sp.size, self.czern.nk), order='F', dtype=np.complex)
        for k in range(self.czern.nk):
            m = self.czern.mtab[k]
            Cnm[:, k] = 2.0*self.czern.coefnorm[k]*((1j)**m)*np.exp(1j*m*ph_sp)
        for k in range(self.czern.nk):
            for f in range(f_sp.size):
                v = Vnm[:, f, k].reshape((Vnm.shape[0], 1))
                c = Cnm[:, k].reshape((1, Cnm.shape[0]))
                Ugrid[:, :, f, k] = np.kron(c, v)
        assert(np.all(np.isfinite(Ugrid)))
        self.Ugrid = Ugrid
        self.Vnm = Vnm
        self.Cnm = Cnm

    def eval_grid(self, beta, i=0, j=0, f_k=0):
        r"""Evaluate the complex point-spread function at a point in a grid.

        The complex point-spread function is

        .. math::

            U(r, \phi, f) = 2 \sum_{n, m} \beta_{n}^{m}
                \sqrt{n + 1} i^{m} V_n^m(r, f) \exp(im\phi).

        See Eq. (4) in [A2015]_.

        Parameters
        ----------
        -   `beta`: `numpy` vector for the Zernike coefficients
            :math:`\beta_n^m`
        -   `i`: `int` first slicing index
        -   `j`: `int` second slicing index
        -   `f_k`: `int` slicing index for the defocus parameter

        Returns
        -------
        -   `U`: complex value of :math:`U(r, \phi, f)`

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

        """
        return np.dot(
            self.Ugrid[i, j, f_k, :],
            self._numpify(beta, np.complex))

    def eval_grid_f(self, beta, f_k=0, const_phase=0.0):
        r"""Slice the complex point-spread function grid at defocus `f_k`.

        The complex point-spread function is

        .. math::

            U(r, \phi, f) = 2 \sum_{n, m} \beta_{n}^{m}
                \sqrt{n + 1} i^{m} V_n^m(r, f) \exp(im\phi).

        See Eq. (4) in [A2015]_.

        Parameters
        ----------
        -   `beta`: `numpy` vector for the Zernike coefficients
            :math:`\beta_n^m`
        -   `f_k`: `int` slicing index for the defocus parameter
        -   `const_phase`: optional `float` to add a constant phase

        Returns
        -------
        -   `U`: A `numpy` array of shape `self.Ugrid.shape[:2]`

        Examples
        --------

        .. code:: python

            import math
            import numpy as np
            import matplotlib.pyplot as p
            import enzpy.enz as enz

            from time import time
            from enzpy.enz import CPsf

            wavelength=632.8e-9    # wavelength in [m]
            aperture_radius=0.002  # aperture radius in [m]
            focal_length=500e-3    # focal length in [m]
            pixel_size=7.4e-6      # pixel size in [m]
            image_width=75         # [pixels], odd to get the origin as well
            image_height=151       # [pixels]
            fspace=np.linspace(    # defocus planes specified by an array of
                -3.0, 2.0, 6)      # defocus parameters
            n_beta=4               # max radial order for the Zernikes

            # compute the diffraction unit (lambda/NA)
            fu = enz.get_field_unit(
                wavelength, aperture_radius, focal_length)

            def make_space(w, p, fu):
                if w % 2 == 0:
                    return np.linspace(-(w/2 - 0.5), w/2 - 0.5, w)*p/fu
                else:
                    return np.linspace(-(w - 1)/2, (w - 1)/2, w)*p/fu

            # image side space
            xspace = make_space(image_width, pixel_size, fu)
            yspace = make_space(image_height, pixel_size, fu)
            image_size = (image_height, image_width)

            # consider complex-valued Zernike polynomials up to the radial
            # order n_beta to approximate the PSF, see Eq. (1) and Eq. (2) in
            # [A2015]_
            cpsf = CPsf(n_beta)

            # make a cartesian grid to evaluate the PSF
            t1 = time()
            cpsf.make_cart_grid(x_sp=xspace, y_sp=yspace, f_sp=fspace)
            t2 = time()
            print('make_cart_grid {:.6f} sec'.format(t2 - t1))

            def plot_psf(U, interpolation='nearest', vmin=None, vmax=None):
                # evaluate the modulus squared
                mypsf = np.square(np.abs(U)).reshape(image_size, order='F')
                p.imshow(
                    mypsf, interpolation=interpolation, vmin=vmin, vmax=vmax,
                    origin='lower')
                p.axis('off')

            def plot_beta_f(beta, fi):
                U = cpsf.eval_grid_f(beta, fi)
                plot_psf(U)

            # beta (diffraction-limited), N_beta = cpsf.czern.nk
            beta = np.zeros(cpsf.czern.nk, dtype=np.complex)
            beta[0] = 1.0

            # or a random beta
            beta = (
                np.random.normal(size=cpsf.czern.nk) +
                1j*np.random.normal(size=cpsf.czern.nk))
            beta[0] = 1.0

            # plot the results
            nn, mm = 2, math.ceil(fspace.size//2)
            p.figure(1)

            for fi, f in enumerate(fspace):
                ax = p.subplot(nn, mm, fi + 1)

                # plot the psf
                plot_beta_f(beta, fi)
                p.colorbar()

                # defocus in rad
                p.title('d={:.1f}'.format(enz.get_defocus(f)))

            p.tight_layout()
            p.show()

        References
        ----------
        ..  [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase
            retrieval for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170
            (2015). `url <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

        """
        return np.exp(-1j*const_phase)*(
            np.dot(
                self.Ugrid[:, :, f_k],
                self._numpify(beta, np.complex)).ravel(order='F'))

    def save(
            self, filename, prepend=None,
            params=HDF5_options, libver='latest'):
        """Save object into an HDF5 file."""
        f = h5py.File(filename, 'w', libver=libver)
        self.save_h5py(f, prepend=prepend, params=params)
        f.close()

    def save_h5py(self, f, prepend=None, params=HDF5_options):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        try:
            params['data'] = self.Ugrid
            f.create_dataset(prefix + 'Ugrid', **params)
        except ValueError:
            pass

        try:
            params['data'] = self.Vnm
            f.create_dataset(prefix + 'Vnm', **params)
        except ValueError:
            pass

        try:
            params['data'] = self.Cnm
            f.create_dataset(prefix + 'Cnm', **params)
        except ValueError:
            pass

        self.czern.save_h5py(f, prepend=prepend)

    @classmethod
    def load(cls, filename, prepend=None):
        """Load object from an HDF5 file."""
        f = h5py.File(filename, 'r')
        z = cls.load_h5py(f, prepend=prepend)
        f.close()

        return z

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""
        z = cls(1)
        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        z.czern = CZern.load_h5py(f, prepend=prepend)

        try:
            z.Ugrid = f[prefix + 'Ugrid'].value
        except ValueError:
            pass
        try:
            z.Vnm = f[prefix + 'Vnm'].value
        except ValueError:
            pass
        try:
            z.Cnm = f[prefix + 'Cnm'].value
        except ValueError:
            pass

        return z
