#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as p

from enzpy.czernike import RZern

"""Example about using enzpy.

Plot the first 10 real-valued Zernike polynomials.

"""


class PhasePlot:

    def __init__(self, n=6, L=200, K=250):
        # real-valued Zernike polynomials up to the n-th radial order
        rzern = RZern(n)

        # make a cartesian grid (unit disk pupil coordinates)
        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        rzern.make_cart_grid(xv, yv)

        self.rzern = rzern
        self.grid_size = (L, K)

    def eval_grid(self, alpha):
        # evaluate the phase on the grid
        Phi = self.rzern.eval_grid(alpha)
        # reshape for plotting
        return Phi.reshape(self.grid_size, order='F')

    def plot_phase(self, Phi, interpolation=None, vmin=None, vmax=None):
        p.imshow(
            Phi, interpolation=interpolation, vmin=vmin, vmax=vmax,
            origin='lower')
        p.axis('off')

    def plot_alpha(self, alpha):
        Phi = self.eval_grid(alpha)
        self.plot_phase(Phi)


if __name__ == '__main__':

    # handy object for plotting
    phaseplot = PhasePlot()

    # vector of real-valued Zernike coefficients
    alpha = np.zeros(phaseplot.rzern.nk)

    p.figure(1)
    for i in range(1, 10):
        # i is Noll's index
        p.subplot(3, 3, i)

        # fill in the Zernike coefficients
        alpha *= 0.0
        alpha[i - 1] = 1.0  # numpy's 0-based indexing

        phaseplot.plot_alpha(alpha)

        # get n and m indices corresponding to the i-th coefficient
        n, m = phaseplot.rzern.noll2nm(i)

        p.title(
            r'$\mathcal{{Z}}_{{{}}}=\mathcal{{Z}}_{{{}}}^{{{}}}$'.format(
                i, n, m))

    p.tight_layout()
    p.show()
