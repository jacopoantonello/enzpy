#!/usr/bin/env python3

# enzpy - Extended Nijboer-Zernike implementation for Python
# Copyright 2016 J. Antonello <jack@antonello.org>
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
import time
import h5py

from enzpy import enz
from enzpy.czernike import RZern, FitZern
from enzpy.enz import CPsf


class Config:

    def __init__(self):
        pass

    def make(
            self, wavelength, aperture_radius, focal_length, pixel_size,
            image_width, image_height,
            n_alpha, n_beta,
            fit_L, fit_K,
            focus_positions):

        self.wavelength = wavelength
        self.aperture_radius = aperture_radius
        self.focal_length = focal_length

        self.image_width = image_width
        self.image_height = image_height
        self.pixel_size = pixel_size

        self.n_alpha, self.n_beta = n_alpha, n_beta

        self.fit_L, self.fit_K = fit_L, fit_K

        self.focus_positions = np.array(focus_positions)

        fu = enz.get_field_unit(
            wavelength,
            aperture_radius, focal_length)

        def make_space(w, p, fu):
            if w % 2 == 0:
                return np.linspace(-(w/2 - 0.5), w/2 - 0.5, w)*p/fu
            else:
                return np.linspace(-(w - 1)/2, (w - 1)/2, w)*p/fu

        # image side space
        xspace = make_space(image_width, pixel_size, fu)
        yspace = make_space(image_height, pixel_size, fu)
        self.xspace, self.yspace = xspace, yspace

        # phase
        self.phase_grid = RZern(n_alpha)
        self.phase_fit = FitZern(self.phase_grid, self.fit_L, self.fit_K)
        print('phase: n_alpha = {}, N_alpha = {}'.format(
            self.n_alpha, self.phase_grid.nk))

        # complex psf
        self.cpsf = CPsf(n_beta)
        self.gpf_fit = FitZern(self.cpsf.czern, self.fit_L, self.fit_K)
        print('cpsf:  n_beta  = {}, N_beta  = {}, N_f = {}'.format(
            n_beta, self.cpsf.czern.nk, self.focus_positions.size))

        # make phase polar grid
        t1 = time.time()
        self.phase_grid.make_pol_grid(
            self.phase_fit.rho_j,
            self.phase_fit.theta_i)
        t2 = time.time()
        print('make phase pol grid {:.6f}'.format(t2 - t1))

        # make gpf polar grid (Zernike approximation)
        t1 = time.time()
        self.cpsf.czern.make_pol_grid(
            self.phase_fit.rho_j,
            self.phase_fit.theta_i)
        t2 = time.time()
        print('make gpf pol grid {:.6f}'.format(t2 - t1))

        # make cpsf cart grid
        t1 = time.time()
        self.cpsf.make_cart_grid(
            x_sp=xspace, y_sp=yspace, f_sp=focus_positions)
        t2 = time.time()
        print('make cpsf cart grid {:.6f}'.format(t2 - t1))

    def save(self, filename, prepend=None, libver='latest'):
        """Save object into an HDF5 file."""
        f = h5py.File(filename, 'w', libver=libver)
        self.save_h5py(f, prepend=prepend)
        f.close()
        print('saved <{}>'.format(filename))

    def save_h5py(self, f, prepend=None):
        """Dump object contents into an opened HDF5 file object."""
        prefix = 'Config/'

        if prepend is not None:
            prefix = prepend + prefix

        params = {
            'chunks': True,
            'shuffle': True,
            'fletcher32': True,
            'compression': 'gzip',
            'compression_opts': 9,
        }

        f.create_dataset(
            prefix + 'wavelength',
            data=np.array([self.wavelength], dtype=np.float))

        f.create_dataset(
            prefix + 'aperture_radius',
            data=np.array([self.aperture_radius], dtype=np.float))

        f.create_dataset(
            prefix + 'focal_length',
            data=np.array([self.focal_length], dtype=np.float))

        f.create_dataset(
            prefix + 'image_width',
            data=np.array([self.image_width], dtype=np.int))

        f.create_dataset(
            prefix + 'image_height',
            data=np.array([self.image_height], dtype=np.int))

        f.create_dataset(
            prefix + 'pixel_size',
            data=np.array([self.pixel_size], dtype=np.float))

        f.create_dataset(
            prefix + 'n_alpha',
            data=np.array([self.n_alpha], dtype=np.int))

        f.create_dataset(
            prefix + 'n_beta',
            data=np.array([self.n_beta], dtype=np.int))

        f.create_dataset(
            prefix + 'fit_L',
            data=np.array([self.fit_L], dtype=np.int))

        f.create_dataset(
            prefix + 'fit_K',
            data=np.array([self.fit_K], dtype=np.int))

        params['data'] = self.focus_positions
        f.create_dataset(prefix + 'focus_positions', **params)

        params['data'] = self.xspace
        f.create_dataset(prefix + 'xspace', **params)

        params['data'] = self.yspace
        f.create_dataset(prefix + 'yspace', **params)

        self.phase_fit.save_h5py(f, prepend=prefix+'phase_fit/')
        self.cpsf.save_h5py(f, prepend=prefix+'cpsf/')
        self.gpf_fit.save_h5py(f, prepend=prefix+'gpf_fit/')

    @classmethod
    def load(cls, filename, prepend=None):
        """Load object from an HDF5 file."""
        f = h5py.File(filename, 'r')
        print('load <{}>'.format(filename))
        z = cls.load_h5py(f, prepend=prepend)
        f.close()

        return z

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""
        sc = cls()
        prefix = 'Config/'

        if prepend is not None:
            prefix = prepend + prefix

        sc.wavelength = float(f[prefix + 'wavelength'].value[0])
        sc.aperture_radius = float(f[prefix + 'aperture_radius'].value[0])
        sc.focal_length = float(f[prefix + 'focal_length'].value[0])
        sc.image_width = int(f[prefix + 'image_width'].value[0])
        sc.image_height = int(f[prefix + 'image_height'].value[0])
        sc.pixel_size = float(f[prefix + 'pixel_size'].value[0])
        sc.n_alpha = int(f[prefix + 'n_alpha'].value[0])
        sc.n_beta = int(f[prefix + 'n_beta'].value[0])
        sc.fit_L = int(f[prefix + 'fit_L'].value[0])
        sc.fit_K = int(f[prefix + 'fit_K'].value[0])

        sc.focus_positions = f[prefix + 'focus_positions'].value
        sc.xspace = f[prefix + 'xspace'].value
        sc.yspace = f[prefix + 'yspace'].value

        sc.phase_fit = FitZern.load_h5py(f, prepend=prefix+'phase_fit/')
        sc.phase_grid = sc.phase_fit.z

        sc.cpsf = CPsf.load_h5py(f, prepend=prefix+'cpsf/')

        sc.gpf_fit = FitZern.load_h5py(f, prepend=prefix+'gpf_fit/')
        sc.gpf_fit.z = sc.cpsf.czern

        print('phase: n_alpha = {}, N_alpha = {}'.format(
            sc.n_alpha, sc.phase_grid.nk))
        print('cpsf:  n_beta  = {}, N_beta  = {},  N_f = {}'.format(
            sc.n_beta, sc.cpsf.czern.nk, sc.focus_positions.size))

        return sc


if __name__ == '__main__':
    import sys
    import os
    import argparse

    from pprint import pprint

    parser = argparse.ArgumentParser(
        description='Make a configuration file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('cfgfile', help='Configuration file name.')

    parser.add_argument(
        '--wavelength', type=float, default=632.8e-9,
        help='Wavelength [m].')
    parser.add_argument(
        '--aperture-radius', type=float, default=0.002,
        help='Aperture radius [m].')
    parser.add_argument(
        '--focal-length', type=float, default=500e-3,
        help='Focal length [m].')
    parser.add_argument(
        '--pixel-size', type=float, default=7.4e-6,
        help='Pixel size [m].')
    parser.add_argument(
        '--image-width', type=int, default=33,
        help='Image width [#pixels].')
    parser.add_argument(
        '--image-height', type=int, default=35,
        help='Image height [#pixels].')
    parser.add_argument(
        '--n-alpha', type=int, default=6,
        metavar='N_ALPHA',
        help='Maximum radial order of the real-valued Zernike polynomials.')
    parser.add_argument(
        '--n-beta', type=int, default=6,
        metavar='N_BETA',
        help='Maximum radial order of the complex-valued Zernike polynomials.')
    parser.add_argument(
        '--focus-positions', type=float, nargs='+',
        default=[0.0, -2.216, 2.768],
        metavar='FP', help='Defocus positions.')
    parser.add_argument(
        '--fit-L', type=int, default=20, metavar='L',
        help='Grid size for the inner products.')
    parser.add_argument(
        '--fit-K', type=int, default=20, metavar='K',
        help='Grid size for the inner products.')
    parser.add_argument(
        '--print', action='store_true',
        help='Print a summary of a saved configuration file.')

    args = parser.parse_args()

    def fillout(a):
        return {
            'wavelength': a.wavelength,
            'aperture_radius': a.aperture_radius,
            'focal_length': a.focal_length,
            'pixel_size': a.pixel_size,
            'image_width': a.image_width,
            'image_height': a.image_height,
            'n_alpha': a.n_alpha,
            'n_beta': a.n_beta,
            'fit_L': a.fit_L,
            'fit_K': a.fit_K,
            'focus_positions': a.focus_positions
        }

    if args.print:
        cfg = Config.load(args.cfgfile)
        params = fillout(cfg)
        params.update({'date': cfg.date.strftime('%c')})
        pprint(params)
        sys.exit(0)

    if os.path.exists(args.cfgfile):
        print('remove old <' + args.cfgfile + '>')
        os.remove(args.cfgfile)

    cfg = Config()
    params = fillout(args)
    cfg.make(**params)
    pprint(params)
    cfg.save(args.cfgfile)
