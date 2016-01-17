#!/usr/bin/python

import sys
import numpy as np
import argparse
import matplotlib.pyplot as p

from numpy.random import normal
from numpy.linalg import norm
from PyQt5 import QtWidgets

from beta_abs import BetaPlot


class Controls(QtWidgets.QWidget):

    def do_cmdline(self, unparsed):
        parser = argparse.ArgumentParser(
            description='''Plot the point-spread function that corresponds to
            a given complex-valued Zernike analysis of the generalised pupil
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

        self.args = parser.parse_args(unparsed[1:])
        return self.args

    def make_gui(self):
        beta = self.beta

        self.setWindowTitle('Complex-valued Zernike coefficients')

        edits1 = list()
        edits2 = list()

        grid = QtWidgets.QGridLayout()

        for i in range(beta.size):
            label = QtWidgets.QLabel(
                '\U0001d4dd<sub>{}</sub>'.format(str(i + 1)))
            label.setStyleSheet('font: 12pt;')
            m = label.fontMetrics()

            edits1.append(QtWidgets.QLineEdit(str(beta[i].real)))
            edits2.append(QtWidgets.QLineEdit(str(beta[i].imag)))

            edits1[i].setFixedSize(m.width('0.000'), m.height() + 2)
            edits2[i].setFixedSize(m.width('0.000'), m.height() + 2)

            grid.addWidget(label, 0, i)
            grid.addWidget(edits1[i], 1, i)
            grid.addWidget(edits2[i], 2, i)

            # edits1[i].editingFinished.connect(self.update_edits)
            # edits2[i].editingFinished.connect(self.update_edits)

        update_button = QtWidgets.QPushButton('u')
        random_button = QtWidgets.QPushButton('r')
        reset_button = QtWidgets.QPushButton('0')

        update_button.setFixedSize(m.width('0.000'), m.height() + 2)
        random_button.setFixedSize(m.width('0.000'), m.height() + 2)
        reset_button.setFixedSize(m.width('0.000'), m.height() + 2)

        grid.addWidget(update_button, 3, i)
        grid.addWidget(random_button, 3, i - 1)
        grid.addWidget(reset_button, 3, i - 2)

        update_button.clicked.connect(self.perform_update)
        random_button.clicked.connect(self.perform_random)
        reset_button.clicked.connect(self.perform_reset)

        self.edits1 = edits1
        self.edits2 = edits2

        self.setLayout(grid)
        self.show()

    def perform_reset(self):
        edits1 = self.edits1
        edits2 = self.edits2
        beta = self.beta

        beta[:] = 0 + 1j*0
        beta[0] = 1

        for i, b in enumerate(beta):
            edits1[i].setText(str(b.real))
            edits2[i].setText(str(b.imag))

        p.ion()
        self.betaplot.plot_beta(beta)
        p.show()

    def perform_random(self):
        edits1 = self.edits1
        edits2 = self.edits2
        beta = self.beta

        beta = normal(
            size=beta.size) + 1j*normal(size=beta.size)
        beta = (self.rms/norm(beta))*beta  # sort of
        beta[0] = 1

        self.beta = beta

        for i, b in enumerate(beta):
            edits1[i].setText(str(b.real))
            edits2[i].setText(str(b.imag))

        p.ion()
        self.betaplot.plot_beta(beta)
        p.show()

    def perform_update(self):
        edits1 = self.edits1
        edits2 = self.edits2
        beta = self.beta

        for i in range(len(edits1)):
            try:
                rl = float(edits1[i].text())
                im = float(edits2[i].text())
            except:
                rl = 0.0
                im = 0.0

            beta[i] = rl + 1j*im
            edits1[i].setText(str(rl))
            edits2[i].setText(str(im))

        p.ion()
        self.betaplot.plot_beta(beta)
        p.show()

    def __init__(self, unparsed):
        super().__init__()

        args = self.do_cmdline(unparsed)

        # plotter
        betaplot = BetaPlot(args)

        # get the complex-valued Zernike polynomials object
        czern = betaplot.psfplot.cpsf.czern

        # beta (diffraction-limited), N_beta = czern.nk
        beta = np.zeros(czern.nk, np.complex)
        beta[0] = 1.0

        # set the beta coefficients randomly
        if args.random:
            beta = normal(size=beta.size) + 1j*normal(size=beta.size)
            beta = (args.rms/norm(beta))*beta  # sort of
            beta[0] = 1

        self.rms = args.rms
        self.beta = beta
        self.betaplot = betaplot

        # make gui
        self.make_gui()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    c = Controls(app.arguments())
    sys.exit(app.exec_())
    del c
