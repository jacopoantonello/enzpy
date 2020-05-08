#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test enz library

"""

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

import logging
import math
import os
import unittest
from tempfile import NamedTemporaryFile
from time import time

import numpy as np
from numpy.linalg import norm
from numpy.random import normal

from enzpy.enz import CPsf


class TestCPsf(unittest.TestCase):

    max_enorm = 1e-9

    def test_eval_U_grid1(self):
        log = logging.getLogger('TestCPsf.test_eval_U_grid1')

        enz = CPsf(8)
        r_sp = float(abs(normal(size=1, ))[0])
        ph_sp = float(normal(size=1, )[0])
        f_sp = float(normal(size=1, )[0])
        enz.make_pol_grid([r_sp], [ph_sp], [f_sp])

        beta = normal(size=enz.czern.nk) + 1j * normal(size=enz.czern.nk)
        a = enz.eval_grid(beta, 0, 0, 0)
        b = enz.U(beta, r_sp, ph_sp, f_sp)

        log.debug('r = {:e}, ph = {:e}, f = {:e}'.format(r_sp, ph_sp, f_sp))
        log.debug('a {:s} {}'.format(str(type(a)), a))
        log.debug('b {:s} {}'.format(str(type(b)), b))
        log.debug('{}'.format(abs(a - b)))

        self.assertTrue(abs(a - b) < self.max_enorm)

    def test_eval_U_grid2(self):
        log = logging.getLogger('TestCPsf.test_eval_U_grid2')

        enz = CPsf(8)
        L, K, P = 50, 70, 3
        A = L * K * P
        r_sp = np.linspace(.01, 2, L)
        ph_sp = [2 * math.pi * i / L for i in range(K)]
        f_sp = [normal(size=1, ) for i in range(P)]
        t1 = time()
        enz.make_pol_grid(r_sp, ph_sp, f_sp)
        t2 = time()
        log.debug('make_pol_grid {:.6f}'.format(t2 - t1))
        self.assertTrue(enz.Ugrid.size == A * enz.czern.nk)

        beta = normal(size=enz.czern.nk) + 1j * normal(size=enz.czern.nk)
        T1 = np.zeros((L, K, P), order='F', dtype=np.complex)
        self.assertTrue(r_sp.size == L)
        self.assertTrue(len(ph_sp) == K)
        self.assertTrue(len(f_sp) == P)
        self.assertTrue(T1.size == L * K * P)
        log.debug(r_sp.size)
        t1 = time()
        for i, r in enumerate(r_sp):
            for j, ph in enumerate(ph_sp):
                for k, f in enumerate(f_sp):
                    T1[i, j, k] = enz.U(beta, r, ph, f)
        t2 = time()
        log.debug('scalar {:.6f}'.format(t2 - t1))

        t1 = time()
        T2 = np.dot(enz.Ugrid, beta)
        t2 = time()
        log.debug('vect {:.6f}'.format(t2 - t1))

        log.debug(enz.Ugrid.flags)
        self.assertTrue(norm(T1.ravel() - T2.ravel()) < self.max_enorm)

    def test_save(self):
        rho_j = np.array([
            0, 0.11111111111111, 0.22222222222222, 0.33333333333333,
            0.44444444444444, 0.55555555555556, 0.66666666666667,
            0.77777777777778, 0.88888888888889, 1
        ])
        theta_i = np.array([
            0, 0.69813170079773, 1.3962634015955, 2.0943951023932,
            2.7925268031909, 3.4906585039887, 4.1887902047864, 4.8869219055841,
            5.5850536063819, 6.2831853071796
        ])
        enz1 = CPsf(8)
        f_sp = np.linspace(-3, 3, 9)
        enz1.make_cart_grid(x_sp=np.linspace(-1, 1, 10),
                            y_sp=np.linspace(-2, 2, 11),
                            f_sp=np.linspace(-3, 3, 12))
        enz1.czern.make_pol_grid(rho_j, theta_i)

        # create tmp path
        tmpfile = NamedTemporaryFile()
        tmppath = tmpfile.name
        tmpfile.close()

        enz1.save(tmppath)
        enz2 = CPsf.load(tmppath)

        for i in range(f_sp.size):
            beta = normal(size=enz1.czern.nk) + 1j * normal(size=enz1.czern.nk)
            self.assertTrue(
                norm(
                    enz1.eval_grid_f(beta, i).ravel() -
                    enz2.eval_grid_f(beta, i).ravel()) < self.max_enorm)

        self.assertTrue(norm(enz1.Ugrid.ravel() - enz2.Ugrid.ravel()) == 0)
        self.assertTrue(norm(enz1.Vnm.ravel() - enz2.Vnm.ravel()) == 0)
        self.assertTrue(norm(enz1.Cnm.ravel() - enz2.Cnm.ravel()) == 0)

        self.assertTrue(norm(enz1.czern.coefnorm - enz2.czern.coefnorm) == 0)
        self.assertTrue(norm(enz1.czern.ntab - enz2.czern.ntab) == 0)
        self.assertTrue(norm(enz1.czern.mtab - enz2.czern.mtab) == 0)
        self.assertTrue(enz1.czern.n == enz2.czern.n)
        self.assertTrue(enz1.czern.nk == enz2.czern.nk)
        self.assertTrue(enz1.czern.normalise == enz2.czern.normalise)
        self.assertTrue(norm(enz1.czern.rhoitab - enz2.czern.rhoitab) == 0)
        self.assertTrue(norm(enz1.czern.rhotab - enz2.czern.rhotab) == 0)
        self.assertTrue(enz1.czern.numpy_dtype == enz2.czern.numpy_dtype)
        self.assertTrue(norm(enz1.czern.ZZ - enz2.czern.ZZ) == 0)

        os.unlink(tmppath)


if __name__ == '__main__':
    unittest.main()
