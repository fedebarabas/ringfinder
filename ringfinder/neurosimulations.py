# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:51:00 2016

@author: Luciano Masullo
@pep8: Federico Barabas
"""

import numpy as np


class sin2D:
    def __init__(self, imSize=100, wvlen=10, theta=15, phase=.25):

        self.imSize = imSize    # image size: n X n
        self.wvlen = wvlen      # wavelength (number of pixels per cycle)
        self.theta = theta      # grating orientation
        self.phase = phase      # phase (0 -> 1)

        self.pi = np.pi

        # X is a vector from 1 to imageSize
        self.X = np.arange(1, self.imSize + 1)
        # rescale X -> -.5 to .5
        self.X0 = (self.X / self.imSize) - .5

        # compute frequency from wavelength
        self.freq = self.imSize/self.wvlen
        # convert to radians: 0 -> 2*pi
        self.phaseRad = (self.phase * 2*self.pi)

        [self.Xm, self.Ym] = np.meshgrid(self.X0, self.X0)     # 2D matrices

        # convert theta (orientation) to radians
        self.thetaRad = (self.theta / 360) * 2*self.pi
        # compute proportion of Xm for given orientation
        self.Xt = self.Xm * np.cos(self.thetaRad)
        # compute proportion of Ym for given orientation
        self.Yt = self.Ym * np.sin(self.thetaRad)
        # sum X and Y components
        self.XYt = np.array(self.Xt + self.Yt)
        # convert to radians and scale by frequency
        self.XYf = self.XYt * self.freq * 2*self.pi

        # make 2D sinewave
        self.sin2d = np.sin(self.XYf + self.phaseRad)

        # display
#        plt.figure()
#        plt.imshow(self.grating, cmap='gray')


class simAxon(sin2D):

    def __init__(self, imSize=50, wvlen=10, theta=15, phase=.25, a=0, b=2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.imSize = imSize       # image size: n X n
        self.wvlen = wvlen         # wavelength (number of pixels per cycle)
        self.theta = theta         # grating orientation
        self.phase = phase         # phase (0 -> 1)

        if b % 2 == 0:
            # sin2D.sin2d squared in order to always get positive values
            self.grating2 = sin2D(self.imSize, 2*self.wvlen, 90 - self.theta,
                                  self.phase).sin2d**b

        else:
            # sin2D.sin2d squared in order to always get positive values
            self.grating2 = sin2D(self.imSize, self.wvlen, 90 - self.theta,
                                  self.phase).sin2d**b

        # Make simulated axon data
        self.data = self.grating2
