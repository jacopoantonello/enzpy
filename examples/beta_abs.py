#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import matplotlib.pyplot as p
import enzpy.enz as enz
import argparse

from numpy.linalg import norm
from numpy.random import normal

from psf_plot import PSFPlot

"""Example about using enzpy.

Plot the point-spread function that corresponds to a given complex-valued
Zernike analysis of the generalised pupil function.

References
----------
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015) . `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

"""


# handy object for plotting the PSF
class BetaPlot:

    def __init__(self, args):
        psfplot = PSFPlot(
                wavelength=args.wavelength,
                aperture_radius=args.aperture_radius,
                focal_length=args.focal_length,
                pixel_size=args.pixel_size,
                image_width=args.image_width,
                image_height=args.image_height,
                fspace=np.linspace(
                    args.defocus_interval[0],
                    args.defocus_interval[1],
                    args.defocus_step),
                n_beta=args.n_beta)
        self.psfplot = psfplot

    def plot_beta(self, beta):
        psfplot = self.psfplot

        # pick the PSF and Zernike objects
        cpsf = psfplot.cpsf
        czern = cpsf.czern

        nn, mm = 2, math.ceil(psfplot.fspace.size//2)
        p.figure(1)
        p.clf()

        for fi, f in enumerate(psfplot.fspace):
            p.subplot(nn, mm, fi + 1)

            # plot the PSF
            psfplot.plot_beta_f(beta, fi)
            p.colorbar()

            # defocus in rad
            p.title('d={:.1f}'.format(enz.get_defocus(f)))

        p.figure(2)
        p.clf()

        # Zernike coefficients of alpha_true and alpha_hat
        p.subplot(3, 1, 1)
        p.plot(range(1, czern.nk + 1), beta.real, marker='o')
        p.ylabel('[rad]')
        p.legend([r'$\beta$ real'])

        p.subplot(3, 1, 2)
        p.plot(range(1, czern.nk + 1), beta.imag, marker='o')
        p.ylabel('[rad]')
        p.legend([r'$\beta$ imag'])

        p.subplot(3, 1, 3)
        p.bar(range(1, czern.nk + 1), np.abs(beta))
        p.xlabel('Noll index')
        p.ylabel('[rad]')
        p.legend([r'$|\beta|$'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Plot the point-spread function that corresponds to a
        given complex-valued Zernike analysis of the generalised pupil
        function.''', epilog='''
        Random beta coefficients: ./beta_abs.py --random.
        Astigmatism, 1 rms rad: ./beta_abs.py --nm 2 2 --rm 1.0.
        Trefoil: ./beta_abs.py --noll 9.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--wavelength', type=float, default=632.8e-9,
        help='Wavelength [m].')
    parser.add_argument(
        '--aperture-radius', type=float, default=0.002,
        help='Aperture radius [m].')
    parser.add_argument(
        '--focal-length', type=float, default=500e-3, help='Focal length [m].')
    parser.add_argument(
        '--image-width', type=int, default=75, help='Image width [#pixels].')
    parser.add_argument(
        '--image-height', type=int, default=151,
        help='Image height [#pixels].')
    parser.add_argument(
        '--pixel-size', type=float, default=7.4e-6, help='Pixel size [m].')
    parser.add_argument(
        '--n-beta', type=int, default=4,
        metavar='N_BETA',
        help='Maximum radial order of the complex-valued Zernike polynomials.')
    parser.add_argument(
        '--defocus-interval', type=float, nargs=2, default=[-3.0, 2.0],
        metavar=('MIN', 'MAX'),
        help='Range of the defocus parameter.')
    parser.add_argument(
        '--defocus-step', type=int, default=6,
        metavar='STEP',
        help='Step size of the defocus parameter.')
    parser.add_argument(
        '--nm', type=int, nargs=2, default=[-1, -1],
        metavar=('N', 'M'),
        help='Specify Zernike polynomial N_n^m using n and m.')
    parser.add_argument(
        '--noll', type=int, default=-1,
        metavar='N_k',
        help="Specify Zernike polynomial N_k using Noll's index k.")
    parser.add_argument(
        '--rms', type=float, default=1.0,
        help='Rms of the beta aberration.')
    parser.add_argument(
        '--random', action='store_true',
        help='Make a random beta aberration.')

    args = parser.parse_args()

    # plotter
    betaplot = BetaPlot(args)

    # get the complex-valued Zernike polynomials object
    czern = betaplot.psfplot.cpsf.czern

    # beta (diffraction-limited), N_beta = czern.nk
    beta = np.zeros(czern.nk, np.complex)
    beta[0] = 1.0

    # nm to linear index conversion
    nmlist = list(zip(czern.ntab, czern.mtab))

    # set a beta coefficient using the (n, m) indeces
    if args.nm[0] != -1 and args.nm[0] != -1:
        try:
            k = nmlist.index(tuple(args.nm))
            beta[k] = args.rms
        except:
            print('Cannot find indeces ' + str(args.nm))
            print('Possible [n, m] indeces are:')
            print(nmlist)
            sys.exit(1)

    # set a beta coefficient using Noll's single index
    if args.noll != -1:
        try:
            beta[args.noll - 1] = args.rms
        except:
            print('Cannot set the required Noll index k.')
            print('Index k must be between 1 and ' + str(beta.size))
            sys.exit(1)

    # set the beta coefficients randomly
    if args.random:
        beta = normal(size=beta.size) + 1j*normal(size=beta.size)
        beta = (args.rms/norm(beta))*beta  # sort of
        beta[0] = 1

    # plot results
    betaplot.plot_beta(beta)
    p.show()
