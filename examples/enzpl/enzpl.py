#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import numpy as np
import math
import h5py

from time import time
from scipy.optimize import fsolve

from cvxopt import lapack, solvers, matrix, blas

from config import Config


def eye(m):
    c = matrix(0.0, (m, m))
    c[::m+1] = 1.0
    return c


def kron(a, b):
    m, n = a.size
    p, q = b.size
    c = matrix(float('nan'), (m*p, n*q))
    for i in range(m):
        for j in range(n):
            c[i*p:(i + 1)*p, j*q:(j + 1)*q] = a[i, j]*b
    return c


def zeros(m, n):
    return matrix(0.0, (m, n))


def mse(x, xhat):
    # TODO check hypotheses in more-1993 (10.1080/10556789308805542)
    # and work out analytically
    a = x.real()
    b = x.imag()

    c = xhat.real()
    d = xhat.imag()

    k = (a.T*a + b.T*b)[0]
    assert(k > 0.0)
    f = zeros(2, 1)
    f[0] = -2*(a.T*c + b.T*d)
    f[1] = -2*(-b.T*c + a.T*d)

    def fw(l):
        w = [0.0, 0.0]
        w[0] = -0.5*f[0]/(k + l)
        w[1] = -0.5*f[1]/(k + l)
        return w

    def fphi(l):
        w = fw(l)
        return w[0]**2 + w[1]**2 - 1

    lopt = fsolve(fphi, -k + 1e-7)
    wopt = fw(lopt)
    copt = matrix(wopt[0][0] + 1j*wopt[1][0])
    msev = (blas.nrm2(copt*x - xhat)**2)/(blas.nrm2(x)**2)
    return copt[0], msev


class ENZPL:

    def __init__(self):
        pass

    def setup(self, AA):
        Nb, Nm = AA.size
        n_x1 = Nb*(Nb + 1)//2
        n_x2 = Nb*(Nb - 1)//2

        print('make Di and Fi...', end='')
        Di = matrix(0.0, (Nb**2, Nm))
        Fi = matrix(0.0, (Nb**2, Nm))
        t1 = time()
        for i in range(Nm):
            d = AA[:, i].real()
            f = AA[:, i].imag()
            Di[:, i] = (d*d.T + f*f.T)[:]
            Fi[:, i] = (f*d.T - d*f.T)[:]
        t2 = time()
        print('{:.6f}'.format(t2 - t1))

        idg = [(i, i) for i in range(Nb)]
        ilw = [(i, j) for i in range(Nb) for j in range(i)]

        print('make E1...', end='')
        t1 = time()
        E1 = matrix(0.0, (Nb**2, n_x1))
        for c, ij in enumerate(idg):
            e = matrix(0.0, (Nb, 1))
            e[ij[0]] = 1.0
            E = e*e.T
            E1[:, c] = E[:]
        for c, ij in enumerate(ilw):
            ei = matrix(0.0, (Nb, 1))
            ej = matrix(0.0, (Nb, 1))
            ei[ij[0]] = 1.0
            ej[ij[1]] = 1.0
            E = ei*ej.T + ej*ei.T
            E1[:, Nb + c] = E[:]
        t2 = time()
        print('{:.6f}'.format(t2 - t1))

        print('make E2...', end='')
        t1 = time()
        E2 = matrix(0.0, (Nb**2, n_x2))
        for c, ij in enumerate(ilw):
            ei = matrix(0.0, (Nb, 1))
            ej = matrix(0.0, (Nb, 1))
            ei[ij[0]] = 1.0
            ej[ij[1]] = 1.0
            E = ei*ej.T - ej*ei.T
            E2[:, c] = E[:]
        t2 = time()
        print('{:.6f}'.format(t2 - t1))

        # min t + lambda*tr(S11)
        n_x = 1 + Nm + n_x1 + n_x2
        c = matrix(0.0, (n_x, 1))
        c[0] = 1.0
        c[1 + Nm:1 + Nm + Nb] = 1.0

        # indeces S >= 0
        print('make indeces...', end='')
        t1 = time()
        I11 = matrix([i + j*2*Nb for j in range(Nb) for i in range(Nb)])
        I12 = matrix([i + (Nb + j)*2*Nb for j in range(Nb) for i in range(Nb)])
        I21 = matrix([Nb + i + j*2*Nb for j in range(Nb) for i in range(Nb)])
        I22 = matrix(
            [Nb + i + (Nb + j)*2*Nb for j in range(Nb) for i in range(Nb)])
        t2 = time()
        print('{:.6f}'.format(t2 - t1))

        dims = {'l': 0, 'q': [1 + Nm], 's': [2*Nb]}

        # Gx + s = h (2nd order cones)
        print('make Gx + s = h...')
        t1 = time()
        G2oc = -eye(1 + Nm)

        # Gx + s = 0 (sdp)
        # S11 = -E1x1, (Nb**2, 1)
        # S21 = -E2x2, (Nb**2, 1)
        # S12 =  E2x2, (Nb**2, 1)
        # S22 = -E1x1, (Nb**2, 1)
        Gtmp = matrix(0.0, ((2*Nb)**2, n_x1 + n_x2))
        # [ -E1   0  0  ]
        # [   0 -E2  0  ]
        # [   0  E2  0  ]
        # [ -E1   0  0  ]
        Gtmp[0*(Nb**2):1*(Nb**2), :n_x1] = -E1
        Gtmp[1*(Nb**2):2*(Nb**2), n_x1:n_x1 + n_x2] = -E2
        Gtmp[2*(Nb**2):3*(Nb**2), n_x1:n_x1 + n_x2] = E2
        Gtmp[3*(Nb**2):4*(Nb**2), :n_x1] = -E1
        T1 = matrix(float('nan'), (2*Nb, 2*Nb))
        tt = matrix([1., 0.])
        bb = matrix([0., 1.])
        T11 = kron(eye(Nb), tt)
        T12 = kron(eye(Nb), bb)
        T1[:2*Nb, :Nb] = T11
        T1[:2*Nb, Nb:] = T12
        T = kron(
            kron(matrix([[1.0, 0.0], [0.0, 0.0]]), T1) +
            kron(matrix([[0.0, 0.0], [0.0, 1.0]]), T1), eye(Nb))
        Gsdp = T*Gtmp

        # G
        G = matrix(0.0, (1 + Nm + (2*Nb)**2, n_x))
        G[:1 + Nm, :1 + Nm] = G2oc
        G[1 + Nm:, 1 + Nm:] = Gsdp

        # h
        h = matrix(0.0, (1 + Nm + (2*Nb)**2, 1))
        # h[1:1 + Nm] = -d # load measurements
        t2 = time()
        print('make Gx + s = h {:.6f}'.format(t2 - t1))

        # A11
        print('make Ax = b...')
        t1 = time()
        A11 = matrix(float('nan'), (Nm, n_x1))
        for i in range(Nm):
            for j in range(n_x1):
                A11[i, j] = Di[:, i].T*E1[:, j]
        assert(not math.isnan(sum(A11)))

        # A12
        A12 = matrix(float('nan'), (Nm, n_x2))
        for i in range(Nm):
            for j in range(n_x2):
                A12[i, j] = -Fi[:, i].T*E2[:, j]
        assert(not math.isnan(sum(A12)))

        A = matrix(0.0, (Nm, n_x))
        A[:, 1:1 + Nm] = eye(Nm)
        A[:, 1 + Nm:1 + Nm + n_x1] = -A11
        A[:, 1 + Nm + n_x1:1 + Nm + n_x1 + n_x2] = A12

        # b
        b = matrix(0.0, (Nm, 1))
        t2 = time()
        print('make Ax = b {:.6f}'.format(t2 - t1))

        self.I11 = I11
        self.I12 = I12
        self.I21 = I21
        self.I22 = I22

        self.Nb = Nb
        self.Nm = Nm
        self.n_x = n_x
        self.G = G
        self.A = A
        self.c = c
        self.h = h
        self.b = b
        self.dims = dims

        self.sA = matrix(0.0 + 1j*0.0, (Nb, Nb))
        self.sW = matrix(0.0, (Nb, 1))
        self.sZ = matrix(0.0 + 1j*0.0, (Nb, Nb))
        self.beta_hat = matrix(0.0 + 1j*0.0, (Nb, 1))

    def save(self, filename, prepend=None, mode='r+', libver='latest'):
        """Save object into an HDF5 file."""
        f = h5py.File(filename, mode=mode, libver=libver)
        self.save_h5py(f, prepend)
        f.close()

    def save_h5py(self, f, prepend=None):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

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
            prefix + 'Nb',
            data=np.array([self.Nb], dtype=np.int))

        f.create_dataset(
            prefix + 'Nm',
            data=np.array([self.Nm], dtype=np.int))

        f.create_dataset(
            prefix + 'n_x',
            data=np.array([self.n_x], dtype=np.int))

        params['data'] = np.array(self.G)
        f.create_dataset(prefix + 'G', **params)

        params['data'] = np.array(self.A)
        f.create_dataset(prefix + 'A', **params)

        params['data'] = np.array(self.c)
        f.create_dataset(prefix + 'c', **params)

        params['data'] = np.array(self.h)
        f.create_dataset(prefix + 'h', **params)

        params['data'] = np.array(self.b)
        f.create_dataset(prefix + 'b', **params)

        params['data'] = np.array(self.sA)
        f.create_dataset(prefix + 'sA', **params)

        params['data'] = np.array(self.sW)
        f.create_dataset(prefix + 'sW', **params)

        params['data'] = np.array(self.sZ)
        f.create_dataset(prefix + 'sZ', **params)

        params['data'] = np.array(self.beta_hat)
        f.create_dataset(prefix + 'beta_hat', **params)

        params['data'] = np.array(self.I11)
        f.create_dataset(prefix + 'I11', **params)

        params['data'] = np.array(self.I12)
        f.create_dataset(prefix + 'I12', **params)

        params['data'] = np.array(self.I21)
        f.create_dataset(prefix + 'I21', **params)

        params['data'] = np.array(self.I22)
        f.create_dataset(prefix + 'I22', **params)

    @classmethod
    def load(cls, filename, prepend=None):
        """Load object from an HDF5 file."""
        f = h5py.File(filename, 'r')
        z = cls.load_h5py(f, prepend=prepend)
        f.close()

        return z

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""
        sc = cls()
        prefix = sc.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        sc.Nb = int(f[prefix + 'Nb'].value[0])
        sc.Nm = int(f[prefix + 'Nm'].value[0])
        sc.n_x = int(f[prefix + 'n_x'].value[0])

        sc.G = matrix(f[prefix + 'G'].value)
        sc.A = matrix(f[prefix + 'A'].value)
        sc.c = matrix(f[prefix + 'c'].value)
        sc.h = matrix(f[prefix + 'h'].value)
        sc.b = matrix(f[prefix + 'b'].value)
        sc.dims = {'l': 0, 'q': [1 + sc.Nm], 's': [2*sc.Nb]}
        sc.sA = matrix(f[prefix + 'sA'].value)
        sc.sW = matrix(f[prefix + 'sW'].value)
        sc.sZ = matrix(f[prefix + 'sZ'].value)
        sc.beta_hat = matrix(f[prefix + 'beta_hat'].value)
        sc.I11 = matrix(f[prefix + 'I11'].value)
        sc.I12 = matrix(f[prefix + 'I12'].value)
        sc.I21 = matrix(f[prefix + 'I21'].value)
        sc.I22 = matrix(f[prefix + 'I22'].value)

        return sc

    def solve_full(self, bi, lmbd, show_progress=True):
        solvers.options['show_progress'] = show_progress

        # load problem data
        self.bi = bi
        self.lmbd = lmbd
        self.c[1 + self.Nm:1 + self.Nm + self.Nb] = lmbd
        self.h[1:1 + self.Nm] = -bi

        t1 = time()
        self.sol = solvers.conelp(
            self.c, self.G, self.h, self.dims, self.A, self.b)
        t2 = time()

        t3 = time()
        if self.sol['status'] in ('optimal'):
            S = self.sol['s'][1 + self.Nm:]
            self.sA[:] = S[self.I11] + 1j*S[self.I21]
            lapack.heevr(
                self.sA, self.sW, jobz='V', range='I', uplo='L',
                vl=0.0, vu=0.0, il=self.Nb, iu=self.Nb, Z=self.sZ)
            self.beta_hat[:] = math.sqrt(self.sW[0])*self.sZ[:, 0]
        else:
            raise RuntimeError('numerical problems')
        t4 = time()

        self.solver_time = t2 - t1
        self.eig_time = t4 - t3

    def solve_brightest(self, bi, lmbd, NmNbeta_ratio, show_progress=True):
        # FIXME
        nbi = np.array(bi).ravel(order='F')

        rind = np.argsort(nbi)[::-1]
        mask = np.zeros(nbi.size)
        mask[rind[:NmNbeta_ratio]] = 1

        linmap = matrix(mask.reshape((mask.size, 1)))
        self.solve_linmap(bi, lmbd, linmap, show_progress)

    def solve_linmap(self, bi, lmbd, linmap, show_progress=True):
        solvers.options['show_progress'] = show_progress

        # save all problem data
        self.bi, self.lmbd = bi, lmbd
        # count alive in map
        Nm2 = int(sum(linmap))
        self.Nm2 = Nm2
        self.linmap = linmap
        n_x1 = self.Nb*(self.Nb + 1)//2
        n_x2 = self.Nb*(self.Nb - 1)//2
        n_x = 1 + Nm2 + n_x1 + n_x2

        c = matrix(0.0, (1 + Nm2 + n_x1 + n_x2, 1))
        c[0] = 1.0
        c[1 + Nm2:1 + Nm2 + self.Nb] = lmbd

        G = matrix(0.0, (1 + Nm2 + (2*self.Nb)**2, n_x))
        G[:1 + Nm2, :1 + Nm2] = -eye(1 + Nm2)
        G[1 + Nm2:, 1 + Nm2:] = self.G[1 + self.Nm:, 1 + self.Nm:]

        h = matrix(0.0, (1 + Nm2 + (2*self.Nb)**2, 1))

        A = matrix(0.0, (Nm2, n_x))
        lbi = matrix(0.0, (Nm2, 1))
        A[:, 1:1 + Nm2] = eye(Nm2)
        pos = 0
        for i in range(self.Nm):
            if linmap[i] != 0.0:
                A[pos, 1 + Nm2:] = self.A[i, 1 + self.Nm:]
                lbi[pos] = bi[i]
                h[1 + pos] = -bi[i]
                pos += 1
        assert(pos == Nm2)

        b = matrix(0.0, (Nm2, 1))

        dims = {'l': 0, 'q': [1 + Nm2], 's': [2*self.Nb]}

        t1 = time()
        self.sol = solvers.conelp(c, G, h, dims, A, b)
        t2 = time()

        self.lc = c
        self.lG = G
        self.lh = h
        self.ldims = dims
        self.lA = A
        self.lb = b
        self.lNm = Nm2
        self.lbi = lbi

        t3 = time()
        if self.sol['status'] in ('optimal', 'unknown'):
            S = self.sol['s'][1 + Nm2:]
            self.sA[:] = S[self.I11] + 1j*S[self.I21]
            lapack.heevr(
                self.sA, self.sW, jobz='V', range='I', uplo='L',
                vl=0.0, vu=0.0, il=self.Nb, iu=self.Nb, Z=self.sZ)
            self.beta_hat[:] = math.sqrt(self.sW[0])*self.sZ[:, 0]
        else:
            raise RuntimeError('numerical problems')
        t4 = time()

        self.solver_time = t2 - t1
        self.eig_time = t4 - t3


if __name__ == '__main__':
    import argparse

    from cmath import exp

    parser = argparse.ArgumentParser(
        description='Create ENZPL data for a given configuration file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'cfgfile', type=argparse.FileType('r'), help='Configuration file')
    parser.add_argument(
        '--solver', choices=['ENZPL'],
        default='ENZPL', help='Solver type')

    args = parser.parse_args()

    # load cfg
    fname = args.cfgfile.name
    args.cfgfile.close()
    args.cfgfile = fname
    cfg = Config.load(args.cfgfile)

    # make mAi matrix
    Nb = cfg.cpsf.czern.nk
    Ni = cfg.xspace.size*cfg.yspace.size
    Nf = cfg.focus_positions.size
    print('Nb = {}, Ni = {}, Nf = {}'.format(Nb, Ni, Nf))
    print('xspace.size = {}, yspace.size = {}'.format(
        cfg.xspace.size,
        cfg.yspace.size))
    mAi = matrix(0.0 + 1j*0.0, (Nb, Ni*Nf))
    for fi in range(Nf):
        fparam = cfg.focus_positions[fi]
        fmul = exp(-1j*fparam)  # redundant
        mAi[:, Ni*fi:Ni*(fi + 1)] = matrix(np.reshape(
            fmul*cfg.cpsf.Ugrid[:, :, fi],
            (Ni, Nb), order='F').transpose().conjugate())

    spl = ENZPL()
    t1 = time()
    spl.setup(mAi)
    t2 = time()
    print('setup ENZPL {:.6f}'.format(t2 - t1))

    print('save {} into <{}>'.format(args.solver, args.cfgfile))
    t1 = time()
    spl.save(args.cfgfile, mode='r+')
    t2 = time()
    print('save {} into <{}> {:.6f}'.format(
        args.solver, args.cfgfile, t2 - t1))
