#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as p
import numpy as np
from numpy.linalg import norm
from numpy.random import normal

from enzpy.czernike import FitZern, RZern
from phase_plot import PhasePlot
"""Example about using enzpy.

Estimate a vector of real-valued Zernike coefficients from a phase grid by
taking inner products.

References
----------
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015) . `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

"""

if __name__ == '__main__':

    # plotting stuff
    phaseplot = PhasePlot()

    # grid sizes
    L, K = 95, 105

    # real-valued Zernike polynomials up to the 6-th radial order
    phase_pol = RZern(6)

    # FitZern computes the approximate inner products, see Eq. (B2) in [A2015]
    ip = FitZern(phase_pol, L, K)
    phase_pol.make_pol_grid(ip.rho_j, ip.theta_i)  # make a polar grid

    # random vector of Zernike coefficients to be estimated
    alpha_true = normal(size=phase_pol.nk)

    # phase grid
    Phi = phase_pol.eval_grid(alpha_true)

    # estimate the random vector from the phase grid
    alpha_hat = ip.fit(Phi)

    # plot the results
    p.figure(2)

    # Zernike coefficients of alpha_true and alpha_hat
    ax = p.subplot2grid((3, 3), (0, 0), colspan=3)
    h1 = ax.plot(range(1, phase_pol.nk + 1), alpha_true, marker='o')
    h2 = ax.plot(range(1, phase_pol.nk + 1), alpha_hat, marker='x')
    p.legend(h1 + h2, [r'$\alpha$', r'$\hat{\alpha}$'])
    p.ylabel('[rad]')
    p.xlabel('$k$')

    # Zernike coefficients error between alpha_true and alpha_hat
    ax = p.subplot2grid((3, 3), (1, 0), colspan=3)
    err = alpha_true - alpha_hat
    ax.bar(range(1, phase_pol.nk + 1), np.abs(err))
    p.title('error {:g} rms rad'.format(norm(err[1:])))
    p.ylabel('[rad]')
    p.xlabel('$k$')

    ax = p.subplot2grid((3, 3), (2, 0))
    phaseplot.plot_alpha(alpha_true)
    p.title(r'$\alpha$')
    p.colorbar()

    ax = p.subplot2grid((3, 3), (2, 1))
    phaseplot.plot_alpha(alpha_hat)
    p.title(r'$\hat{\alpha}$')
    p.colorbar()

    ax = p.subplot2grid((3, 3), (2, 2))
    phaseplot.plot_alpha(alpha_true - alpha_hat)
    p.title(r'$\alpha - \hat{\alpha}$')
    p.colorbar()

    p.tight_layout()
    p.show()
