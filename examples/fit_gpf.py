#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as p

from numpy.linalg import norm
from numpy.random import normal
from enzpy.czernike import CZern, FitZern

"""Example about using enzpy.

Estimate a vector of complex-valued Zernike coefficients from a grid by taking
inner products numerically. The Zernike coefficients can be used to approximate
the generalised pupil function.

References
----------
 .. [A2015] Jacopo Antonello and Michel Verhaegen, "Modal-based phase retrieval
    for adaptive optics," J. Opt. Soc. Am. A 32, 1160-1170 (2015) . `url
    <http://dx.doi.org/10.1364/JOSAA.32.001160>`__.

"""

if __name__ == '__main__':

    # grid sizes
    L, K = 95, 105

    # complex-valued Zernike polynomials up to the 4-th radial order
    gpf_pol = CZern(4)  # to approximate the GPF

    # FitZern computes the approximate inner products, see Eq. (B4) in [A2015]
    ip = FitZern(gpf_pol, L, K)
    gpf_pol.make_pol_grid(ip.rho_j, ip.theta_i)  # make a polar grid

    # random vector of Zernike coefficients to be estimated
    beta_true = normal(size=gpf_pol.nk) + 1j*normal(size=gpf_pol.nk)

    # random generalised pupil function P
    P = gpf_pol.eval_grid(beta_true)

    # estimate the random vector from the GPF grid
    beta_hat = ip.fit(P)

    # plot the results
    p.figure(2)

    # real part of the Zernike coefficients of beta_true and beta_hat
    ax = p.subplot(3, 1, 1)
    h1 = ax.plot(range(1, gpf_pol.nk + 1), beta_true.real, marker='o')
    h2 = ax.plot(range(1, gpf_pol.nk + 1), beta_hat.real, marker='x')
    p.legend(
        h1 + h2, [r'$\mathcal{R}[\beta]$', r'$\mathcal{R}[\hat{\beta}]$'])
    p.ylabel('[rad]')
    p.xlabel('$k$')

    # imaginary part of the Zernike coefficients of beta_true and beta_hat
    ax = p.subplot(3, 1, 2)
    h1 = ax.plot(range(1, gpf_pol.nk + 1), beta_true.imag, marker='o')
    h2 = ax.plot(range(1, gpf_pol.nk + 1), beta_hat.imag, marker='x')
    p.legend(
        h1 + h2, [r'$\mathcal{I}[\beta]$', r'$\mathcal{I}[\hat{\beta}]$'])
    p.ylabel('[rad]')
    p.xlabel('$k$')

    # Zernike coefficients error between beta_true and beta_hat
    ax = p.subplot(3, 1, 3)
    err = beta_true - beta_hat
    ax.bar(range(1, gpf_pol.nk + 1), np.abs(err))
    err = beta_true - beta_hat
    p.title('error {:g} rms rad'.format(norm(err[1:])))
    p.ylabel('[rad]')
    p.xlabel('$k$')

    p.tight_layout()
    p.show()
