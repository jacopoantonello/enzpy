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

from time import time
from numpy.random import normal

from enzpy import enz


class OptSys:

    def __init__(self, cfg):
        self.cfg = cfg
        assert(self.cfg.focus_positions[0] == 0.0)

        self.alpha_ab = np.zeros(self.cfg.phase_grid.nk)
        self.alpha_dm = np.zeros(self.cfg.phase_grid.nk)

    def reset(self):
        self.alpha_ab[:] = 0
        self.alpha_dm[:] = 0

    def set_alpha_ab(self, alpha):
        self.alpha_ab[:] = alpha

    def measure_at_defocus(self, fi):
        """Apply the fi-th defocus with the DM and perform the measurements.

        Returns (PSF, SH, alpha_residual, u_real, ts).
        PSF: measured / simulated PSF sub image
        SH: measured full SH image / zero
        alpha_residual: SH estimated residual aberration / alpha_ab + alpha_dm
        u_real: DM control signal including offset / zero
        ts: timestamp
        """
        self.alpha_dm[:] = 0
        self.alpha_dm[3] = enz.get_defocus(self.cfg.focus_positions[fi])

    def apply_correction(self, alpha_hat):
        """Apply the aberration correction with the DM.

        - Apply the correction -alpha_hat with the DM.
        - Perform the measurements.

        Returns (PSF, SH, alpha_residual, u_real, ts).
        PSF: measured / simulated PSF sub image
        SH: measured full SH image / zero
        alpha_residual: SH estimated residual aberration / alpha_ab + alpha_dm
        u_real: DM control signal including offset / zero
        ts: timestamp
        """

        self.alpha_dm[:] = -alpha_hat


class SimOptSys(OptSys):

    def __init__(self, cfg, sh_image_shape, u_size, noise_std=0.0):
        super().__init__(cfg)

        self.alpha_res = np.zeros(self.cfg.phase_grid.nk)
        self.zeros_sh = np.zeros(sh_image_shape, order='F')
        self.zeros_u = np.zeros(u_size)
        self.noise_std = noise_std

        self.set_alpha_ab(self.alpha_res)

    def set_alpha_ab(self, alpha):
        super().set_alpha_ab(alpha)

        # TODO remove allocation
        phase_ab = self.cfg.phase_grid.eval_grid(self.alpha_ab)
        gpf = np.exp(1j*phase_ab)

        # save the beta corresponding to this alpha aberration
        self.beta_ab = self.cfg.gpf_fit.fit(gpf)

    def measure_at_defocus(self, fi, exact_data_gen=False):
        super().measure_at_defocus(fi)

        # compute the residual aberration
        self.alpha_res[:] = self.alpha_ab + self.alpha_dm

        # TODO remove allocation

        if exact_data_gen:
            # EXACT
            # compute the psf at the fi-th defocus, using beta_ab
            # the defocusing is included in the U_grid
            psf = np.square(np.abs(self.cfg.cpsf.eval_grid_f(
                self.beta_ab, fi, self.cfg.focus_positions[fi]))).reshape(
                    self.cfg.image_height, self.cfg.image_width, order='F')
        else:
            # APPROXIMATE
            # compute the residual aberration
            self.alpha_res[:] = self.alpha_ab + self.alpha_dm

            # eval the residual aberration
            gpf = np.exp(1j*self.cfg.phase_grid.eval_grid(self.alpha_res))

            # get the ENZ modes for the residual aberration
            self.beta_res = self.cfg.gpf_fit.fit(gpf)

            psf = np.square(np.abs(self.cfg.cpsf.eval_grid_f(
                self.beta_res, 0))).reshape(
                    self.cfg.image_height, self.cfg.image_width, order='F')

        # add noise
        if self.noise_std > 0.0:
            psf += normal(scale=self.noise_std, size=psf.shape)
            psf[psf < 0.0] = 0.0

        return {
            'psf': psf,
            'sh': self.zeros_sh,
            'alpha_res': self.alpha_res.copy(),
            'u_real': self.zeros_u,
            'ts': time()
            }

    def apply_correction(self, beta_hat):
        super().apply_correction(beta_hat)

        # compute the residual aberration
        self.alpha_res[:] = self.alpha_ab + self.alpha_dm

        # eval the residual aberration
        gpf = np.exp(1j*self.cfg.phase_grid.eval_grid(self.alpha_res))

        # get the ENZ modes for the residual aberration
        self.beta_res = self.cfg.gpf_fit.fit(gpf)

        psf = np.square(np.abs(self.cfg.cpsf.eval_grid_f(
            self.beta_res, 0))).reshape(
                self.cfg.image_height, self.cfg.image_width, order='F')

        # add noise
        if self.noise_std > 0.0:
            psf += normal(scale=self.noise_std, size=psf.shape)

        return {
            'psf': psf,
            'sh': self.zeros_sh,
            'alpha_res': self.alpha_res.copy(),
            'u_real': self.zeros_u,
            'ts': time(),
            }
