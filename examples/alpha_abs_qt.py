#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
import matplotlib.pyplot as p

from numpy.random import normal
from numpy.linalg import norm
from PyQt5 import QtWidgets

from enzpy.czernike import RZern, CZern, FitZern

from beta_abs import BetaPlot
from phase_plot import PhasePlot


class Controls(QtWidgets.QWidget):

    def do_cmdline(self, unparsed):
        parser = argparse.ArgumentParser(
            description='''Plot the point-spread function that corresponds to
            a given real-valued Zernike analysis of the phase aberration
            function. The Zernike coefficients can be adjusted with a Qt
            widget.''',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
            '--image-width', type=int, default=75,
            help='Image width [#pixels].')
        parser.add_argument(
            '--image-height', type=int, default=151,
            help='Image height [#pixels].')
        parser.add_argument(
            '--pixel-size', type=float, default=7.4e-6, help='Pixel size [m].')
        parser.add_argument(
            '--n-alpha', type=int, default=4,
            metavar='N_ALPHA',
            help=(
                'Maximum radial order of the real-valued Zernike' +
                'polynomials.'))
        parser.add_argument(
            '--n-beta', type=int, default=4,
            metavar='N_BETA',
            help=(
                'Maximum radial order of the complex-valued Zernike' +
                'polynomials.'))
        parser.add_argument(
            '--defocus-interval', type=float, nargs=2, default=[-3.0, 2.0],
            metavar=('MIN', 'MAX'),
            help='Range of the defocus parameter.')
        parser.add_argument(
            '--defocus-step', type=int, default=6,
            metavar='STEP',
            help='Step size of the defocus parameter.')
        parser.add_argument(
            '--rms', type=float, default=1.0,
            help='Rms of the beta aberration.')
        parser.add_argument(
            '--random', action='store_true',
            help='Make a random beta aberration.')
        parser.add_argument(
            '--fit-L', type=int, default=95, metavar='L',
            help='Grid size for the inner products.')
        parser.add_argument(
            '--fit-K', type=int, default=105, metavar='K',
            help='Grid size for the inner products.')

        self.args = parser.parse_args(unparsed[1:])
        return self.args

    def make_gui(self):
        alpha = self.alpha

        self.setWindowTitle('Real-valued Zernike coefficients')

        edits1 = list()
        grid = QtWidgets.QGridLayout()

        for i in range(alpha.size):
            label = QtWidgets.QLabel(
                '\U0001d4e9<sub>{}</sub>'.format(str(i + 1)))
            label.setStyleSheet('font: 12pt;')
            m = label.fontMetrics()

            edits1.append(QtWidgets.QLineEdit(str(alpha[i])))
            edits1[i].setFixedSize(m.width('0.000'), m.height() + 2)

            grid.addWidget(label, 0, i)
            grid.addWidget(edits1[i], 1, i)

        update_button = QtWidgets.QPushButton('u')
        random_button = QtWidgets.QPushButton('r')
        reset_button = QtWidgets.QPushButton('0')

        update_button.setFixedSize(m.width('0.000'), m.height() + 2)
        random_button.setFixedSize(m.width('0.000'), m.height() + 2)
        reset_button.setFixedSize(m.width('0.000'), m.height() + 2)

        grid.addWidget(update_button, 2, i)
        grid.addWidget(random_button, 2, i - 1)
        grid.addWidget(reset_button, 2, i - 2)

        update_button.clicked.connect(self.perform_update)
        random_button.clicked.connect(self.perform_random)
        reset_button.clicked.connect(self.perform_reset)

        self.edits1 = edits1

        self.setLayout(grid)
        self.show()

    def show_plot(self):
        self.alpha2beta()

        p.ion()
        p.figure(10)
        p.clf()

        # plot alpha
        p.subplot2grid((1, 3), (0, 0), colspan=2)
        h1 = p.plot(range(1, self.phase_pol.nk + 1), self.alpha, marker='o')
        p.legend(h1, [r'$\alpha$'])
        p.ylabel('[rad]')
        p.xlabel('$k$')
        p.subplot2grid((1, 3), (0, 2))
        self.phaseplot.plot_alpha(self.alpha)
        p.title(r'$\alpha$')
        p.colorbar()

        # plot beta
        self.betaplot.plot_beta(self.beta_hat)
        p.show()

    def perform_reset(self):
        edits1 = self.edits1
        alpha = self.alpha

        alpha[:] = 0

        for i, a in enumerate(alpha):
            edits1[i].setText(str(a))

        self.show_plot()

    def perform_random(self):
        edits1 = self.edits1
        alpha = self.alpha

        alpha1 = normal(size=alpha.size-1)
        alpha1 = (self.rms/norm(alpha1))*alpha1
        alpha[1:] = alpha1
        del alpha1

        for i, a in enumerate(alpha):
            edits1[i].setText(str(a))

        self.show_plot()

    def perform_update(self):
        edits1 = self.edits1
        alpha = self.alpha

        for i in range(len(edits1)):
            try:
                a = float(edits1[i].text())
            except BaseException:
                a = 0.0

            alpha[i] = a
            edits1[i].setText(str(a))

        self.show_plot()

    def alpha2beta(self):
        # evaluate the phase corresponding to alpha
        Phi = self.phase_pol.eval_grid(self.alpha)

        # evaluate the generalised pupil function P corresponding to alpha
        P = np.exp(1j*Phi)

        # estimate the beta coefficients from P
        self.beta_hat = self.ip.fit(P)

    def __init__(self, unparsed):
        super().__init__()

        args = self.do_cmdline(unparsed)

        # plot objects
        phaseplot = PhasePlot(n=args.n_alpha)  # to plot beta and the PSF
        betaplot = BetaPlot(args)  # to plot the phase

        # complex-valued Zernike polynomials for the GPF
        ip = FitZern(CZern(args.n_beta), args.fit_L, args.fit_K)

        # real-valued Zernike polynomials for the phase
        phase_pol = RZern(args.n_alpha)
        phase_pol.make_pol_grid(ip.rho_j, ip.theta_i)  # make a polar grid

        # real-valued Zernike coefficients
        alpha = np.zeros(phase_pol.nk)

        # set the alpha coefficients randomly
        if args.random:
            alpha1 = normal(size=alpha.size-1)
            alpha1 = (args.rms/norm(alpha1))*alpha1
            alpha[1:] = alpha1
            del alpha1

        self.rms = args.rms
        self.alpha = alpha
        self.phase_pol = phase_pol
        self.ip = ip

        self.betaplot = betaplot
        self.phaseplot = phaseplot

        # fit beta coefficients from alpha coefficients
        self.alpha2beta()

        # make gui
        self.make_gui()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    c = Controls(app.arguments())
    sys.exit(app.exec_())
    del c
