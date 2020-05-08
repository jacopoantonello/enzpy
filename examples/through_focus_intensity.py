#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi, sqrt
from time import time

import matplotlib.pyplot as p
import numpy as np

from enzpy.enz import CPsf
"""Example about using enzpy.

This example is taken from [W1]_, and computes the intensity of the
point-spread function as a function of the radial coordinate and the defocus
parameter.

Note that the V_n^m functions in [W1]_ and in enzpy are defined differently,
due to a factor epsilon_m and to the scaling in the r coordinate.

References:
----------
 .. [W1] http://www.nijboerzernike.nl/_downloads/example.m

"""

if __name__ == '__main__':
    NA = 0.50  # numerical aperture
    diam = 0.25  # diameter [um]
    wavelength = 0.200  # lambda [um]
    eps = 0.00001  # offset to avoid division by zero

    radius = diam / 2
    ap = 2 * pi * (NA / wavelength) * radius
    d = 1 / 8 * ap**2 + 1 / 384 * ap**4 + 1 / 10240 * ap**6  # optimal d. Why!?
    scale_z = wavelength / (2 * pi) * 1 / (1 - sqrt(1 - NA**2))

    rspace = (np.arange(0, 0.5, 0.05) + eps) * NA / wavelength
    fspace = np.arange(-2, 2, 0.05) / scale_z + 1j * d

    cpsf = CPsf(0)  # aberration free, only n = m = 0

    t1 = time()
    field1 = cpsf.V(n=0, m=0, r=rspace, f=fspace)
    t2 = time()
    print('field1 {:.6f} sec'.format(t2 - t1))

    field1 = np.squeeze(field1)
    I1 = np.square(np.abs(field1))
    I1 *= (1 / I1.max())

    ff, rr = np.meshgrid(np.real(fspace), rspace)
    levels = np.sort(
        np.concatenate((np.array([0.025, 0.05]), np.arange(0, 1, 0.1))))
    c = p.contour(ff, rr, I1, levels)
    p.clabel(c, inline=1)
    p.xlabel('Focus [um]')
    p.ylabel('Radius [um]')
    p.title('Aberration free PSF, diameter = {} [um]'.format(diam))
    p.show()
