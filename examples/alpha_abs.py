#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as p
import argparse

from numpy.linalg import norm
from numpy.random import normal

from enzpy.czernike import RZern, CZern, FitZern

from beta_abs import BetaPlot
from phase_plot import PhasePlot

"""Example about using enzpy.

Plot the point-spread function that corresponds to a given real-valued Zernike
analysis of the phase aberration function.

References
----------
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015) . `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Plot the point-spread function that corresponds to a
        given real-valued Zernike analysis of the phase aberration
        function.''', epilog='''
        Random alpha coefficients: ./alpha_abs.py --random.
        Astigmatism, 1 rms rad: ./alpha_abs.py --nm 2 2 --rm 1.0.
        Trefoil: ./alpha_abs.py --noll 9.
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
        '--n-alpha', type=int, default=4,
        metavar='N_ALPHA',
        help='Maximum radial order of the real-valued Zernike polynomials.')
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
        help='Rms of the alpha aberration.')
    parser.add_argument(
        '--random', action='store_true',
        help='Make a random alpha aberration.')
    parser.add_argument(
        '--fit-L', type=int, default=95, metavar='L',
        help='Grid size for the inner products.')
    parser.add_argument(
        '--fit-K', type=int, default=105, metavar='K',
        help='Grid size for the inner products.')

    args = parser.parse_args()

    # complex-valued Zernike polynomials for the GPF
    ip = FitZern(CZern(args.n_beta), args.fit_L, args.fit_K)

    # real-valued Zernike polynomials for the phase
    phase_pol = RZern(args.n_alpha)
    phase_pol.make_pol_grid(ip.rho_j, ip.theta_i)  # make a polar grid

    # real-valued Zernike coefficients
    alpha = np.zeros(phase_pol.nk)

    # nm to linear index conversion
    nmlist = list(zip(phase_pol.ntab, phase_pol.mtab))

    # set an alpha coefficient using the (n, m) indeces
    if args.nm[0] != -1 and args.nm[0] != -1:
        try:
            k = nmlist.index(tuple(args.nm))
            alpha[k] = args.rms
        except:
            print('Cannot find indeces ' + str(args.nm))
            print('Possible [n, m] indeces are:')
            print(nmlist)
            sys.exit(1)

    # set an alpha coefficient using Noll's single index
    if args.noll != -1:
        try:
            alpha[args.noll - 1] = args.rms
        except:
            print('Cannot set the required Noll index k.')
            print('Index k must be between 1 and ' + str(alpha.size))
            sys.exit(1)

    # set the alpha coefficients randomly
    if args.random:
        alpha1 = normal(size=alpha.size-1)
        alpha1 = (args.rms/norm(alpha1))*alpha1
        alpha[1:] = alpha1
        del alpha1

    # evaluate the phase corresponding to alpha
    Phi = phase_pol.eval_grid(alpha)

    # evaluate the generalised pupil function P corresponding to alpha
    P = np.exp(1j*Phi)

    # estimate the beta coefficients from P
    beta_hat = ip.fit(P)

    # plot the results
    phaseplot = PhasePlot(n=args.n_alpha)  # to plot beta and the PSF
    betaplot = BetaPlot(args)  # to plot the phase

    # plot alpha
    p.figure(10)
    p.subplot2grid((1, 3), (0, 0), colspan=2)
    h1 = p.plot(range(1, phase_pol.nk + 1), alpha, marker='o')
    p.legend(h1, [r'$\alpha$'])
    p.ylabel('[rad]')
    p.xlabel('$k$')
    p.subplot2grid((1, 3), (0, 2))
    phaseplot.plot_alpha(alpha)
    p.title(r'$\alpha$')
    p.colorbar()

    # plot beta
    betaplot.plot_beta(beta_hat)

    p.show()
