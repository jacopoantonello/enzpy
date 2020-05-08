#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse
from time import time

import matplotlib.pyplot as p
import numpy as np
from numpy.linalg import norm
from numpy.random import normal
from skimage.restoration import unwrap_phase

from config import Config
from cvxopt import matrix
from enzpl import ENZPL, mse
from optsys import SimOptSys

parser = argparse.ArgumentParser(
    description='Run a single random aberration correction experiment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfgfile',
                    type=argparse.FileType('r'),
                    help='configuration file')
parser.add_argument('--pl-NmNbeta-ratio',
                    type=float,
                    default=10.0,
                    metavar='P',
                    help='Use only the brightest P*N_beta of pixels for PL.')
parser.add_argument('--rms',
                    type=float,
                    default=0.8,
                    help='rms value of the aberration.')
parser.add_argument('--pl-lmbd',
                    type=float,
                    default=1.0,
                    help='Optimisation lambda.')
parser.add_argument('--pl-solver',
                    choices=['ENZPL', 'CustomENZPL'],
                    default='ENZPL',
                    help='solver type')
algorithms = ['enzpl', 'enzap', 'fftap']
parser.add_argument('--alg', choices=algorithms, default=algorithms[0])
parser.add_argument('--plot-steps', action='store_true')

args = parser.parse_args()

print('load cfg... ')
t1 = time()
fname = args.cfgfile.name
args.cfgfile.close()
args.cfgfile = fname
cfg = Config.load(args.cfgfile)
t2 = time()
print('load cfg {:.6f}'.format(t2 - t1))

print('load {} from <{}>...'.format(args.pl_solver, args.cfgfile))
t1 = time()
if args.pl_solver == 'ENZPL':
    spl = ENZPL.load(args.cfgfile)
t2 = time()
print('load {} from <{}> {:.6f}...'.format(args.pl_solver, args.cfgfile,
                                           t2 - t1))

# instance simulation class (no Shack-Hartmann, no DM)
optsys = SimOptSys(cfg, (2, 2), 17)
print('Run simulation')

# make aberration
nk = cfg.phase_grid.nk
alpha_true = np.zeros(nk)
randn = normal(size=5)
alpha_true[4:9] = (args.rms / norm(randn)) * randn

# apply aberration
optsys.set_alpha_ab(alpha_true)

Ni = cfg.xspace.size * cfg.yspace.size
Nf = cfg.focus_positions.size
mbi = matrix(0.0, (Ni * Nf, 1))
mbi_sim = matrix(0.0, (Ni * Nf, 1))
mr = list()

# loop through focus planes
meas_list = list()
for fi in range(Nf):
    mr.append(optsys.measure_at_defocus(fi))

    pr = mr[-1]['psf']

    print('focus {:+e} [{:.4f}, {:.4f}]'.format(
        cfg.focus_positions[fi],
        pr.min(),
        pr.max(),
    ))
    mbi[fi * Ni:(fi + 1) * Ni] = pr.ravel(order='F')
    meas_list.append(pr.ravel(order='F'))


def parse_enz_solution(cfg, betak):
    if betak.ndim == 2:
        gpf = cfg.cpsf.czern.eval_grid(betak[:, 0])
    else:
        gpf = cfg.cpsf.czern.eval_grid(betak)
    wrph = np.arctan2(gpf.imag, gpf.real)
    wrph = np.mod(wrph, 2 * np.pi) - np.pi

    ut1 = time()
    unph = unwrap_phase(wrph.reshape((cfg.fit_L, cfg.fit_K),
                                     order='F')).ravel(order='F')
    ut2 = time()

    ft1 = time()
    alpha_hat = cfg.phase_fit.fit(unph)
    ft2 = time()

    alpha_hat[0] = 0.0

    return alpha_hat, ft2 - ft1, ut2 - ut1, wrph, unph


def enzpl():
    print('enzppl: lambda {:e} N_m/N_beta {:.4f}'.format(
        args.pl_lmbd, args.pl_NmNbeta_ratio))
    if args.pl_NmNbeta_ratio > 0.0:
        spl.solve_brightest(mbi,
                            args.pl_lmbd,
                            NmNbeta_ratio=round(args.pl_NmNbeta_ratio *
                                                cfg.cpsf.czern.nk),
                            show_progress=True)
    else:
        spl.solve_full(mbi, args.pl_lmbd, show_progress=True)
    beta_hat = np.array(spl.beta_hat).ravel()

    # parse solution
    alpha_hat, dtfit, dtunwrap, wrph, unph = parse_enz_solution(cfg, beta_hat)

    print('enzppl: *******************************************************')
    print('enzppl: RESULTS')
    mbeta_ab = matrix(optsys.beta_ab)
    copt, msev = mse(mbeta_ab, spl.beta_hat)
    print('enzppl: abs(copt) {:e}'.format(abs(copt)))
    print('enzppl: copt {0.real:e} {0.imag:+e}i'.format(copt))
    print('enzppl: msev {:e} '.format(msev))
    er = norm(alpha_true[1:] - alpha_hat[1:])
    rer = er / norm(alpha_true[1:])
    print('enzppl: residual rms {:e} rer {:.3f}'.format(er, rer))
    print('enzppl: solver_time: {:.6f}'.format(spl.solver_time))
    print('enzppl: eig_time: {:.6f}'.format(spl.eig_time))
    print('enzppl: *******************************************************')

    return alpha_hat


# run the algorithm
if args.alg == 'enzpl':
    alpha_hat = enzpl()

# apply correction
cr = optsys.apply_correction(alpha_hat)

pr = cr['psf']
print('corr [{:.4f}, {:.4f}]'.format(cfg.focus_positions[fi], pr.min(),
                                     pr.max()))


def plot_step(r):
    mr = r['psf']
    p.subplot(2, 1, 1)
    p.imshow(mr, interpolation='none')
    p.colorbar()
    p.title('max real {:.3f}'.format(mr.max()))

    p.subplot(2, 1, 2)
    h1, = p.plot(range(1, alpha_true.size + 1), r['alpha_res'], marker='x')
    p.ylim([-args.rms, args.rms])
    p.grid()
    nr = norm(r['alpha_res'])
    p.title('real {:.3f}'.format(nr))


if len(mr) == 3:
    if args.plot_steps:
        for fi in range(Nf):
            p.figure(fi)
            plot_step(mr[fi])

        p.figure(fi + 1)
        plot_step(cr)

    mmin = [a['alpha_res'].min() for a in mr] + [cr['alpha_res'].min()]
    mmax = [a['alpha_res'].max() for a in mr] + [cr['alpha_res'].max()]
    mmin = min(mmin) - .1
    mmax = max(mmax) + .1

    p.figure()
    p.subplot(2, 2, 1)
    h = p.plot(range(1, alpha_true.size + 1), mr[0]['alpha_res'])
    p.grid()
    p.ylim([mmin, mmax])
    p.legend(['rms {:.3f}'.format(norm(mr[0]['alpha_res']))])
    p.xlabel(r'$\mathcal{Z}_i$')
    p.ylabel('[rad]')
    p.title('initial aberration')

    p.subplot(2, 2, 2)
    p.plot(range(1, alpha_true.size + 1), mr[1]['alpha_res'])
    p.grid()
    p.ylim([mmin, mmax])
    p.legend(['rms {:.3f}'.format(norm(mr[1]['alpha_res']))])
    p.xlabel(r'$\mathcal{Z}_i$')
    p.ylabel('[rad]')
    p.title('second measurement')

    p.subplot(2, 2, 3)
    p.plot(range(1, alpha_true.size + 1), mr[2]['alpha_res'])
    p.grid()
    p.ylim([mmin, mmax])
    p.legend(['rms {:.3f}'.format(norm(mr[2]['alpha_res']))])
    p.xlabel(r'$\mathcal{Z}_i$')
    p.ylabel('[rad]')
    p.title('third measurement')

    p.subplot(2, 2, 4)
    p.plot(range(1, alpha_true.size + 1), cr['alpha_res'])
    p.grid()
    p.ylim([mmin, mmax])
    p.legend(['rms {:.3f}'.format(norm(cr['alpha_res']))])
    p.xlabel(r'$\mathcal{Z}_i$')
    p.ylabel('[rad]')
    p.title('aberration correction')

    p.tight_layout()

    def myplot(myim):
        p.imshow(myim, interpolation='none')
        p.colorbar()
        p.axis('off')

    p.figure()
    p.subplot(2, 2, 1)
    myplot(mr[0]['psf'])
    p.title('initial aberration')

    p.subplot(2, 2, 2)
    myplot(mr[1]['psf'])
    p.title('second measurement')

    p.subplot(2, 2, 3)
    myplot(mr[2]['psf'])
    p.title('third measurement')

    p.subplot(2, 2, 4)
    myplot(cr['psf'])
    p.title('aberration correction')

    p.show()
