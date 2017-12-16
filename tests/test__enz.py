#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Extended Nijboer-Zernike toolbox.

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

import unittest
import h5py

from numpy.linalg import norm

from enzpy._enz import vnmpocnp


class Test__Enz(unittest.TestCase):

    max_enorm = 1e-9

    def test_vnmpocnp(self):
        data = h5py.File('refVnmpo.h5', 'r')

        ref = data['ref'].value
        r = data['r'].value
        f = data['f'].value
        n = data['n'].value
        m = data['m'].value

        data.close()

        out = vnmpocnp(r, f, n, m)
        self.assertTrue(out.shape[0] == r.size)
        self.assertTrue(out.shape[1] == r.size)
        self.assertTrue(out.shape[2] == n.size)
        err = norm((out - ref).ravel())
        self.assertTrue(err < self.max_enorm)

if __name__ == '__main__':
    unittest.main()
