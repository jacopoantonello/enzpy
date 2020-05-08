#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from time import time

import matplotlib.pyplot as p
import numpy as np

import enzpy.enz as enz
from enzpy.enz import CPsf
"""Example about using enzpy.

Plots a diffraction-limited point-spread function at different defocus planes.

References
----------
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015) . `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

"""


class PSFPlot:
    def __init__(
            self,
            wavelength=632.8e-9,  # wavelength in [m]
            aperture_radius=0.002,  # aperture radius in [m]
            focal_length=500e-3,  # focal length in [m]
            pixel_size=7.4e-6,  # pixel size in [m]
            image_width=75,  # [pixels], odd to get the origin as well
            image_height=151,  # [pixels]
            fspace=np.linspace(  # defocus planes specified by an array of
                -3.0, 2.0, 6),  # defocus parameters
            n_beta=4  # max radial order for the Zernikes
    ):

        # compute the diffraction unit (lambda/NA)
        fu = enz.get_field_unit(wavelength=wavelength,
                                aperture_radius=aperture_radius,
                                exit_pupil_sphere_radius=focal_length)

        def make_space(w, p, fu):
            if w % 2 == 0:
                return np.linspace(-(w / 2 - 0.5), w / 2 - 0.5, w) * p / fu
            else:
                return np.linspace(-(w - 1) / 2, (w - 1) / 2, w) * p / fu

        # image side space
        xspace = make_space(image_width, pixel_size, fu)
        yspace = make_space(image_height, pixel_size, fu)

        # consider complex-valued Zernike polynomials up to the radial order
        # n_beta to approximate the PSF, see Eq. (1) and Eq. (2) in [A2015]
        cpsf = CPsf(n_beta)

        # make a cartesian grid to evaluate the PSF
        t1 = time()
        cpsf.make_cart_grid(x_sp=xspace, y_sp=yspace, f_sp=fspace)
        t2 = time()
        print('make_cart_grid {:.6f} sec'.format(t2 - t1))

        self.xspace = xspace
        self.yspace = yspace
        self.fspace = fspace

        self.cpsf = cpsf
        self.image_size = (image_height, image_width)

    def eval_grid_f(self, beta, fi):
        # evaluate the complex point-spread function at the fi-th defocus plane
        U = self.cpsf.eval_grid_f(beta, fi)
        # reshape for plotting
        return U.reshape(self.image_size, order='F')

    def plot_psf(self, U, interpolation='nearest', vmin=None, vmax=None):
        # evaluate the modulus squared
        mypsf = np.square(np.abs(U))
        p.imshow(mypsf,
                 interpolation=interpolation,
                 vmin=vmin,
                 vmax=vmax,
                 origin='lower')
        p.axis('off')

    def plot_beta_f(self, beta, fi):
        U = self.eval_grid_f(beta, fi)
        self.plot_psf(U)


if __name__ == '__main__':

    # handy object for plotting the PSF
    psfplot = PSFPlot()

    # beta (diffraction-limited), N_beta = cpsf.czern.nk
    beta = np.zeros(psfplot.cpsf.czern.nk, dtype=np.complex)
    beta[0] = 1.0
    beta[5] = 1j * 0.3
    beta[6] = -1j * 0.3

    # plot the results
    nn, mm = 2, math.ceil(psfplot.fspace.size // 2)
    p.figure(1)

    for fi, f in enumerate(psfplot.fspace):
        ax = p.subplot(nn, mm, fi + 1)

        # plot the psf
        psfplot.plot_beta_f(beta, fi)
        p.colorbar()

        # defocus in rad
        p.title('d={:.1f}'.format(enz.get_defocus(f)))

    p.tight_layout()
    p.show()
